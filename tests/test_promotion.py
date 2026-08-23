"""The promotion contract — PLAN.md §25 and §26's ranked successor #1.

A promotion is a one-file change to `best_solution.json`, and this repo has now twice found
that it is not: §16 and §19 each moved the shipped genome and left references behind that
nothing checked.

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
SHIPPED_GENOME_HASH = "09e8188"
SHIPPED_PROMOTED_IN = "PLAN.md §26 (2026-08-14)"

# The genome §14's constants were measured on.  `study_svk_rescore.run_control` is pinned to
# this FILE rather than to the shipped pointer, which is §25's fix; if the file itself ever
# moves, that control silently changes meaning and this catches it.
CONTROL_GENOME_FILE = "stage3_minwall_best_1.2.json"
CONTROL_GENOME_HASH = "350f4c7"

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
      3. DRIVERS THAT PAIR `best_solution.json` WITH A GENOME-SPECIFIC CONSTANT.  Known one:
         `studies/study_svk_rescore.py`'s §14 control, which is now pinned to
         {CONTROL_GENOME_FILE} by file for this reason (PLAN §25).  If you add another
         constant measured on one wheel, pin it to a FILE, never to the shipped pointer.
      4. `make svk` — the feasibility gate.  It is the check that runs about once a
         promotion, so assume it has rotted since you last looked.
      5. PRESERVE THE OUTGOING GENOME under its own name, and leave the `note` field in
         `best_solution.json` saying where the new one came from.
      6. LEAVE `tests/test_golden.py` READING {GOLDEN_GENOME_FILE} — §10's decoupling is what
         makes a promotion unable to re-baseline the regression net.
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
