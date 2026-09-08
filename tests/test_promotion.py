"""The promotion contract — PLAN.md §25 and §26's ranked successor #1.

A promotion is a one-file change to `best_solution.json`, and this repo has now three times
found that it is not: §16, §19 and §115 each moved the shipped genome and left references
behind that nothing checked.

  PLAN §25   `study_svk_rescore.py`'s §14 control read `best_solution.json` while comparing
             against a constant measured on `350f4c7`.  §19 made those different wheels and
             the control went red on a CORRECT change.  Nobody saw it, because the same
             driver had also been failing a term-set guard since §18 — it is the promotion
             gate, so it runs about once a promotion, which is the worst possible moment to
             discover it is broken.
  PLAN §26   PLAN.md's own top-of-file banner still declared `350f4c7` shipped and said in
             terms that the shipped genome had not changed.  False since §16, through two
             promotions, in the one place `SVK_PLAN.md` step 7 requires to be amended when
             the genome moves.
  PLAN §117  THE CHECKLIST BELOW WAS WALKED IN FULL AND THE SUITE WAS STILL 62 RED, because
             items 1-6 do not include "re-run the suite against the new genome" or
             "regenerate the artifacts that read it".  That is item 7, added at §118 — the
             one this file learned about itself rather than about a driver.
  PLAN §133  ITEM 7 IS NECESSARY AND IT IS NOT SUFFICIENT.  §115's promotion predates item
             7, so nothing walked it here; §132 finally did what it asks — ran the suite
             against the new genome — and §133 classified what came back.  Ten of the
             eleven survivors were tests pinned to a genome-SPECIFIC quantity through the
             shipped pointer, which is item 3's defect in the half of the tree item 3 does
             not mention.  That is item 9.  Item 7 tells you WHICH tests went red; it does
             not tell you whether a red means the wheel changed or the test was only ever a
             claim about one wheel, and those call for opposite repairs.

WHAT THIS FILE DOES NOT TRY TO DO.  There are ~100 references to `best_solution.json` across
`src/`, `studies/` and `tests/`, and almost all of them are correct: they mean "the design we
ship" and following a promotion is exactly right.  The defect is the narrow case — a
genome-SPECIFIC constant sitting next to a read of a file that MOVES — and no grep can tell
those two apart.  So this does not scan.  It puts a tripwire on the promotion itself: the
shipped hash is recorded here, and moving it turns this file red with the checklist in the
failure message.  The point is not the assertion, it is the checklist arriving at the moment
someone is promoting rather than months later.

These are file-level checks: no FEA, no meshing, no CAD.  They cost milliseconds and they run
on every `make test`.
"""

import hashlib
import json
import os

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------------------
# THE SHIPPED GENOME.  Update this in the SAME change that updates `best_solution.json`, and
# do the checklist in the failure message below while you are here.
SHIPPED_GENOME_HASH = "b729e86"
SHIPPED_PROMOTED_IN = "PLAN.md §115 (2026-09-06)"

# The genome §14's constants were measured on.  `study_svk_rescore.run_control` is pinned to
# this FILE rather than to the shipped pointer, which is §25's fix; if the file itself ever
# moves, that control silently changes meaning and this catches it.
CONTROL_GENOME_FILE = "stage3_minwall_best_1.2.json"
CONTROL_GENOME_HASH = "350f4c7"

# The genome the FILLET arc's published constants were measured on — FILLET_PLAN.md PART 3,
# PART 5, PART 13 and PART 20, and PLAN §37/§51, all dated 2026-08-17 to 2026-08-23, when
# this was what `best_solution.json` held.  §120 pinned three drivers to this FILE for the
# same reason §25 pinned `run_control` to the one above, so it is now load-bearing in the
# same way: overwrite it and six recorded constants silently change what they are about.
# It is also §115's checklist item 5 — the preserved outgoing genome — so the two roles
# are one file and neither may move it.
PROFILE_GENOME_FILE = "stage3_knee_best_medium.json"
PROFILE_GENOME_HASH = "09e8188"

# The GA/beam reference the regression net is pinned to.  §10 decoupled the golden test from
# the shipped genome precisely so a promotion cannot re-baseline it.
GOLDEN_GENOME_FILE = "best_solution_ga_beam.json"
GOLDEN_GENOME_HASH = "36aed36"

PROMOTION_CHECKLIST = f"""
    The shipped genome moved.  That is a normal thing to do — but it is NOT a one-file
    change, and this repo has twice proved it by finding the leftovers weeks later.  Before
    updating SHIPPED_GENOME_HASH in this file, walk the list:

      1. PLAN.md's TOP-OF-FILE BANNER.  Add the new genome to the chain and scope any
         sentence that claims what ships today.  SVK_PLAN.md step 7 requires this and it was
         skipped at §16 and §19 (PLAN §26).
      2. `export/wheel.step` AND its manifest — rebuild with `make export`, so the shipped
         STEP does not describe a previous genome.  That silent failure is what
         `wheel_step_export.py` was audited for.
      3. DRIVERS THAT PAIR `best_solution.json` WITH A GENOME-SPECIFIC CONSTANT.  FOUR
         KNOWN, and the last three were found by §119 walking item 7:
           - `studies/study_svk_rescore.py`'s §14 control, pinned to {CONTROL_GENOME_FILE}
             by file for this reason (PLAN §25) — the worked precedent for the fix.
           - `studies/study_fillet_fold.py`, whose reconciliation checks FILLET_PLAN.md
             PART 3 / PART 5 against a sweep at the shipped genome.  8 of 8 rows agree on
             `09e8188`, 4 of 8 on `b729e86`, same code.
           - `studies/study_tri_block.py` (`control_reproduces_the_collapse`,
             `shipped_control_is_the_published_0.78`).
           - `studies/study_fillet_block.py` (PART 13's argmax, PART 20's bisections, and
             both candidate-constant surfaces).
         ALL FOUR ARE NOW PINNED BY FILE — the last three at §120, to
         {PROFILE_GENOME_FILE}.  Until then each PROMOTION MADE THEM EXIT 1 and a re-run
         filed a saved failure rather than a refresh, which is why item 7 was not
         sufficient on its own.  If you add another constant measured on one wheel, pin it
         to a FILE, never to the shipped pointer.

         AND PIN THE WHOLE FAMILY, NOT THE CELLS THAT WENT RED.  §120 pinned
         `study_fillet_block`'s four failing checks and left `cliff_profile` following the
         shipped pointer; the two are compared against EACH OTHER, so the table came back
         reading 0.7909 against a 0.5520 bound — coherent-looking, every field present,
         and describing two different wheels.  A half-pinned comparison is worse than an
         unpinned one, because nothing goes red.
      4. `make svk` — the feasibility gate.  It is the check that runs about once a
         promotion, so assume it has rotted since you last looked.
      5. PRESERVE THE OUTGOING GENOME under its own name, and leave the `note` field in
         `best_solution.json` saying where the new one came from.
      6. LEAVE `tests/test_golden.py` READING {GOLDEN_GENOME_FILE} — §10's decoupling is what
         makes a promotion unable to re-baseline the regression net.
      7. RE-RUN THE SUITE *AFTER* WRITING THE NEW GENOME, AND RE-RUN EVERY DRIVER THAT
         DEFAULTS TO `best_solution.json`, COMMITTING ITS ARTIFACT.  Items 1-6 imply neither,
         and §115 walked all six and still left the suite 62 red (PLAN §117): nine of those
         say a committed `studies/*.json` no longer describes this tree, and TWO are strict
         XPASSes — an `xfail` whose own reason names a promotion as its clearing condition is
         a FAILURE the day it clears, not a bonus.  A suite run from before the genome swap
         does not answer this, which is how §115's record came to say `895 passed / 0 failed`
         for a commit that is red.
      8. AND THE ARTIFACTS THAT READ *THOSE* ARTIFACTS — item 7 one level down, added by
         §120 after refreshing three of them.  A driver's output is another driver's input:
         `study_tri_bend.py` and `study_tri_rule.py` both take `sweep.best`'s cell from
         `studies/study_tri_block.json`, and `study_fillet_kt.py` and
         `study_fillet_wiring.py` read the corner and junction artifacts.  None of those
         re-derives what it reads, so a promotion invalidates them WITHOUT running them and
         without any test noticing — `study_tri_rule.json` had no check at all, and
         `tests/test_fillet_artifact_chain.py` was written because neither fillet consumer
         did either.  Re-run the consumers after the producers, in that order.
      9. AND THE SAME RULE FOR *TESTS*, WHICH IS ITEM 3 ONE NEIGHBOURHOOD OVER AND WAS
         MISSING UNTIL §133.  Item 3 says: a constant measured on one wheel gets pinned to a
         FILE, never to the shipped pointer.  Twenty-one test files read
         `best_solution.json` off disk, and the rule had never been stated for them.  §133
         walked §117's eleven survivors and found TEN OF THE ELEVEN were tests asserting a
         genome-SPECIFIC quantity while reading the shipped pointer — the exact defect item
         3 exists to prevent, in the other half of the tree.  They are not eleven bugs: all
         eleven pass at HEAD's code with the outgoing genome swapped in.
         SO BEFORE PROMOTING, FOR EACH TEST THAT GOES RED, ASK WHAT IT IS A CLAIM ABOUT:
           - about the MECHANISM ("SVK mis-recovery is dangerous", "the max diverges while
             the p99 converges", "no threshold can sit on this statistic") — then its
             fixture is a genome CHOSEN to exhibit that, pinned by FILE, and the promotion
             should never have touched it.
           - about THE WHEEL THAT SHIPS — then the promotion changed the answer, and the
             docstring's numbers are what needs updating, deliberately, with the new
             measurement.
         THESE ARE OPPOSITE REPAIRS AND THE WRONG ONE GOES GREEN WHILE ASSERTING NOTHING.
         Two of §133's ten say so in their own assertion text — "this test can no longer
         tell the two apart and is not guarding anything" — which is what a symptom-pin
         sounds like the day its symptom moves.  And a demonstration whose vehicle stops
         demonstrating does not fail loudly: `test_wheel_fea.py`'s
         `test_peak_stress_diverges_but_the_field_converges` has now been miscalibrated by
         TWO CONSECUTIVE promotions, and its own docstring diagnoses the first correctly.
         THE FIDELITY IS PART OF THE CLAIM TOO.  §133's eleventh red raises
         `NewtonDivergedError` rather than asserting, which reads as the shipped wheel
         having no equilibrium path; its fixture is `CFG = "smoke"`, and at `coarse` the
         same genome converges with a continuation spread of 3.775e-15.  A fixture cheap
         enough to run on every promotion is a fixture that can condemn a design it cannot
         resolve.
"""


def _hash(genes):
    """`wheel_step_export.genome_hash`, duplicated on purpose: this file must not import the
    CAD env's module to answer a question about a JSON file."""
    canon = json.dumps({k: round(float(v), 12) for k, v in sorted(genes.items())})
    return hashlib.sha256(canon.encode()).hexdigest()[:7]


def _genome(name):
    with open(os.path.join(HERE, name)) as fh:
        return json.load(fh)


def test_the_shipped_genome_is_the_one_this_tree_documents():
    """THE TRIPWIRE.  Everything else in this file is a consequence of it."""
    got = _hash(_genome("best_solution.json")["genes"])
    assert got == SHIPPED_GENOME_HASH, (
        f"best_solution.json is `{got}`, this tree documents `{SHIPPED_GENOME_HASH}` "
        f"(promoted in {SHIPPED_PROMOTED_IN}).\n{PROMOTION_CHECKLIST}")


def test_the_shipped_step_is_not_older_than_the_shipped_genome():
    """`wheel.step` silently describing a previous genome is the bug the exporter was audited
    for; its own `warn_if_stale` only fires when the exporter is RUN, which is exactly when it
    is already being fixed.

    THE HASH HALF OF THIS IS ALREADY COVERED by `test_golden.py::test_genome_hash_matches_
    manifest`, which asserts manifest-vs-shipped directly. What is NOT covered there is
    STALENESS: that test compares two files to each other, so a manifest rebuilt from an old
    genome file agrees with itself. This adds the clock and the `source` field.
    """
    man_path = os.path.join(HERE, "export/wheel_step_manifest.json")
    if not os.path.exists(man_path):
        pytest.skip("no export/wheel_step_manifest.json in this tree")
    man = json.load(open(man_path))
    assert man["source"] == "best_solution.json", (
        f"the shipped manifest was built from {man['source']}, not best_solution.json — "
        f"`make export` with EXPORT_GENOME set writes under its own stem for this reason")

    step = os.path.join(HERE, "export/wheel.step")
    if not os.path.exists(step):
        pytest.skip("no export/wheel.step in this tree")
    genome_mtime = os.path.getmtime(os.path.join(HERE, "best_solution.json"))
    assert os.path.getmtime(step) >= genome_mtime - 1.0, (
        "export/wheel.step is OLDER than best_solution.json — the shipped STEP describes a "
        f"genome that is no longer shipped. Run `make export`.\n{PROMOTION_CHECKLIST}")

    step = os.path.join(HERE, "export/wheel.step")
    if os.path.exists(step):
        genome_mtime = os.path.getmtime(os.path.join(HERE, "best_solution.json"))
        assert os.path.getmtime(step) >= genome_mtime - 1.0, (
            "export/wheel.step is OLDER than best_solution.json — the shipped STEP describes "
            "a genome that is no longer shipped. Run `make export`.")


def test_the_control_genome_has_not_moved():
    """PLAN §25's fix pins the §14 control to a file. A file can be overwritten too."""
    got = _hash(_genome(CONTROL_GENOME_FILE)["genes"])
    assert got == CONTROL_GENOME_HASH, (
        f"{CONTROL_GENOME_FILE} is `{got}`, not `{CONTROL_GENOME_HASH}` — "
        "`study_svk_rescore.run_control` compares this genome against PLAN §14 constants "
        "measured ON it, so moving the file silently changes what the control means.")


def test_the_fillet_arcs_constant_genome_has_not_moved():
    """§120's fix pins three drivers to a file, which makes that file's CONTENT load-bearing.

    The same failure mode as `test_the_control_genome_has_not_moved` one arc over: six
    constants across FILLET_PLAN PART 3 / PART 5 / PART 13 / PART 20 and PLAN §37/§51 are
    compared against surfaces measured on this genome, so overwriting the file would leave
    every self-check still computed, still green, and no longer about anything recorded.
    """
    got = _hash(_genome(PROFILE_GENOME_FILE)["genes"])
    assert got == PROFILE_GENOME_HASH, (
        f"{PROFILE_GENOME_FILE} is `{got}`, not `{PROFILE_GENOME_HASH}` — "
        "`study_fillet_fold`, `study_tri_block` and `study_fillet_block` compare "
        "FILLET_PLAN and PLAN constants against surfaces measured ON this genome, so "
        "moving the file silently changes what six recorded numbers are about.")


def test_the_golden_reference_genome_has_not_moved():
    """§10 pointed the regression net at a genome under its own name so that promoting cannot
    re-baseline it. This pins the file's CONTENT.

    IT DOES NOT ASSERT THAT `test_golden.py` NEVER READS `best_solution.json`, and the first
    draft of this test did — wrongly. That file reads the shipped genome in exactly one
    place, `test_genome_hash_matches_manifest`, and its docstring argues at length why that
    is correct: a manifest hash is a statement about whichever genome the exporter last ran
    on, so reading the pinned fixture there would turn a traceability check into a second
    copy of the fixture. §10 decoupled the FIXTURE, not the file. Scanning the file is the
    wrong instrument for the question.
    """
    got = _hash(_genome(GOLDEN_GENOME_FILE)["genes"])
    assert got == GOLDEN_GENOME_HASH, (
        f"{GOLDEN_GENOME_FILE} is `{got}`, not `{GOLDEN_GENOME_HASH}` — the regression net is "
        "pinned to this genome so that promoting cannot re-baseline it. If this moved, every "
        f"drift `test_golden.py` exists to catch has just been re-baselined.\n{PROMOTION_CHECKLIST}")


def test_the_outgoing_genome_was_preserved_under_its_own_name():
    """A promotion overwrites the only copy of the previous shipped genome unless someone
    kept one. Every promotion so far has; this asserts the habit rather than trusting it."""
    shipped = _genome("best_solution.json")
    note = shipped.get("note", "")
    assert note, (
        "best_solution.json has no `note` recording where it came from. The field existed at "
        "§13 and was lost at §19; it is the only in-file provenance the shipped genome "
        f"carries.\n{PROMOTION_CHECKLIST}")

    hashes = set()
    for name in os.listdir(HERE):
        if not name.endswith(".json") or name == "best_solution.json":
            continue
        try:
            rec = json.load(open(os.path.join(HERE, name)))
        except (ValueError, OSError):
            continue
        if isinstance(rec, dict) and isinstance(rec.get("genes"), dict):
            hashes.add(_hash(rec["genes"]))
    assert SHIPPED_GENOME_HASH in hashes, (
        f"no file other than best_solution.json holds `{SHIPPED_GENOME_HASH}` — the genome "
        "the descent produced should survive under its own name, so the shipped file is a "
        "copy rather than the only original.")
