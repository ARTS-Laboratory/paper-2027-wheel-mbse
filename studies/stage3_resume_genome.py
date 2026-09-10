"""Warm-restart a Stage-3 descent from the trajectory it has already written.

    .venv-opt/bin/python studies/stage3_resume_genome.py stage3_run.json
                                                       (make stage3-resume TRAJ=stage3_run.json)

WHY THIS EXISTS. `wheel_stage3` persists its trajectory after every step, and does it
atomically, precisely so that "a run measured in tens of minutes must survive a kill"
(`_persist`/`_write`, wheel_stage3.py:875-896 — tmp file plus `os.replace`, so a kill
mid-write cannot truncate it). The genome does not get the same treatment: `--best-out` is
written exactly once, by `wg.save_record`, AFTER the descent loop returns
(wheel_stage3.py:1299-1310). A run killed at step 299 of 300 therefore leaves a complete
step-by-step history and NO promotable genome at all.

The history is not usable as a restart either, because the two ends disagree on shape:
`load_genes` reads a TOP-LEVEL `"genes"` (wheel_stage3.py:980-982) and a trajectory nests
its genes under `best` and `final`. `--genome <trajectory>` raises `KeyError`. **This driver
is that adapter and nothing else.** It reads one JSON and writes one JSON; it solves nothing,
imports no jax, and peaks around 30 MB — so it stays runnable on a box whose memory has just
been taken by whatever killed the descent.

**IT IS A WARM RESTART, NOT A RESUME, AND THE DIFFERENCE IS THREE CONCRETE THINGS.** It
recovers the descent's POSITION. It cannot recover its momentum:

  1. Adam's moments are re-zeroed on entry to every `descend` call (wheel_stage3.py:589-590)
     and `_row` never records them (`:740-762` writes z, grad, lr, terms, report). Neither is
     the secant warm-start `delta0`, the rejection counter `n_reject`, or the halved `lr`
     that a rejected step leaves behind (`:666`).
  2. `--lr` cannot restore a decayed rate. It sets `lr0`, and the schedule is then recomputed
     from scratch — `lr_t = cosine_lr(lr, i-1, steps)` (`:595`, `:258-263`). Reproducing the
     rate that step 82 of 300 was running at needs `--steps 218` together with an `lr0` the
     cosine's own shape cannot exactly produce. The restart command printed below therefore
     carries the ORIGINAL `lr0` and the REMAINING step count, which restarts the cosine at
     full amplitude on a shorter horizon. That is an approximation, stated rather than hidden.
  3. Four run-shaping parameters are not in `settings` at all (`--n-sub`, `--grad-clip`,
     `--max-rejects`, `--t1-reject`), so a restart can only assume they took their defaults.
     The printed command names them explicitly for that reason: if the killed run set one by
     hand, the command is wrong until a human edits it, and it is better for that to be
     visible than inferred.

So the restarted descent is not bit-identical to the run it continues, and a record that
quotes both halves as one descent is quoting two. Expect a transient while Adam re-estimates
its moments.

THE OUTPUT'S KEY ORDER IS LOAD-BEARING, WHICH IS NOT OBVIOUS. `genome_hash` is
order-INDEPENDENT — it hashes `sorted(genes.items())` (wheel_genome.py:122-132). `load_genes`
is order-DEPENDENT — `list(json.load(fh)["genes"].values())`, with no name lookup anywhere
(wheel_stage3.py:980-982). A genome file written in sorted key order would therefore carry
the CORRECT hash, satisfy every promotion check, and hand a scrambled 14-vector to the
descent. This driver never sorts and never rebuilds the dict: it passes `best.genes` through
verbatim, because `json.load` preserves file order and a trajectory's own `genes` blocks are
already written in `GENE_NAMES` order by `wg.vector_to_genes` (wheel_stage3.py:854, :867).
`tests/test_genome_key_order.py` is the regression guard on that reasoning.

Writing goes through `wheel_genome.save_record` (wheel_genome.py:148-172) rather than a bare
`json.dump` for the same reason `study_fillet_pnorm_box.write_genome_files` uses the record
shape: it enforces "exactly the 14 canonical keys and nothing else" and hands back the hash,
which this driver then checks against the hash the trajectory recorded for the same iterate.
That check is free and it is the only thing standing between a silent adapter bug and a
day-long descent restarted from the wrong point.
"""

import argparse
import json
import os

import project_paths as PP
import wheel_genome as WG

# A genome path handed to `--genome` is resolved against the repo root, not the caller's cwd
# (`load_genes` -> `os.path.join(HERE, path)` with `HERE = PP.ROOT`, wheel_stage3.py:129,
# :981).  Writing anywhere else produces a file that reads fine here and is invisible there.
HERE = PP.ROOT

# `settings` key -> the flag that reproduces it.  `steps` is deliberately absent: a restart
# wants the REMAINING count, not the original one, and that is computed below.
SETTING_FLAGS = (("config", "--config"),
                 ("optimizer", "--optimizer"),
                 ("phase_scheme", "--phase-scheme"),
                 ("n_phase", "--n-phase"),
                 ("seed", "--seed"),
                 ("kinematics", "--kinematics"),
                 ("min_wall_mm", "--min-wall"),
                 ("workers", "--workers"),
                 ("fidelity_check_every", "--fidelity-check-every"))

# Run-shaping flags no `settings` block records, with the defaults a restart must assume.
# Named explicitly in the printed command so an operator can see what is being assumed.
UNRECORDED_DEFAULTS = (("--n-sub", 8), ("--grad-clip", 1.0), ("--max-rejects", 3),
                       ("--t1-reject", 1.0e4))


def read_trajectory(path):
    """The trajectory, plus the two derived numbers a restart needs.

    Reading a trajectory that is still being written is safe by construction — `_write`
    swaps it into place with `os.replace` (wheel_stage3.py:894), so a reader sees either the
    previous complete file or the next one, never a partial one.
    """
    with open(path) as fh:
        doc = json.load(fh)
    if "steps" not in doc or not doc["steps"]:
        raise SystemExit(
            f"{path} carries no `steps` trace — it is a `*_best*`/`*_check*` summary, not a "
            f"descent trajectory, and there is no iterate in it to restart from.")
    settings = doc.get("settings", {})
    done = len(doc["steps"])
    requested = settings.get("steps")
    remaining = None if requested is None else max(0, int(requested) - done)
    return doc, done, remaining


def restart_command(settings, genome_path, remaining, out_name, lr0):
    """The command that continues the run, as a list of argv words.

    `--out` is forced to a NEW name: `_write` would otherwise overwrite the trajectory this
    genome was just recovered from (wheel_stage3.py:889-896), destroying the history in the
    act of resuming it.
    """
    # A relative `--genome` is resolved against the repo root, so a genome written there can
    # be named by its basename and one written anywhere else must be named absolutely.
    named = os.path.basename(genome_path) \
        if os.path.dirname(os.path.abspath(genome_path)) == os.path.abspath(HERE) \
        else os.path.abspath(genome_path)
    argv = [".venv-opt/bin/python", "-u", "src/wheel_stage3.py",
            "--start", "best", "--genome", named]
    for key, flag in SETTING_FLAGS:
        if settings.get(key) is not None:
            argv += [flag, str(settings[key])]
    if remaining is not None:
        argv += ["--steps", str(remaining)]
    if lr0 is not None:
        argv += ["--lr", repr(lr0)]
    for flag, default in UNRECORDED_DEFAULTS:
        argv += [flag, str(default)]
    argv += ["--out", out_name, "--best-out", out_name.replace(".json", "_best.json")]
    return argv


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("trajectory", help="a stage3_*.json written by wheel_stage3 --out")
    ap.add_argument("--from", dest="which", choices=("best", "final"), default="best",
                    help="which iterate to restart from (default: best — the one "
                         "`--best-out` would have promoted had the run finished)")
    ap.add_argument("--out", default=None,
                    help="genome file to write (default: <trajectory>_resume.json, at the "
                         "repo root where --genome looks for it)")
    args = ap.parse_args()

    traj_path = args.trajectory if os.path.isabs(args.trajectory) \
        else os.path.join(HERE, args.trajectory)
    doc, done, remaining = read_trajectory(traj_path)

    block = doc.get(args.which)
    if not block or "genes" not in block:
        raise SystemExit(f"{args.trajectory} has no `{args.which}.genes` block.")

    stem = os.path.basename(traj_path)[:-len(".json")] if traj_path.endswith(".json") \
        else os.path.basename(traj_path)
    out = args.out or (stem + "_resume.json")
    out_path = out if os.path.isabs(out) else os.path.join(HERE, out)
    if os.path.abspath(out_path) == os.path.abspath(traj_path):
        raise SystemExit("refusing to write the genome over the trajectory it came from.")

    settings = doc.get("settings", {})
    # `lr0` is not in `settings`; step 0's recorded rate is `cosine_lr(lr0, -1, steps)`, which
    # is within 6e-5 of lr0 for any horizon this tree runs, so it is the honest recoverable
    # value.  See the module docstring on why restoring the DECAYED rate is not possible.
    lr0 = doc["steps"][0].get("lr")

    written_hash = WG.save_record(
        out_path, block["genes"],
        source="studies/stage3_resume_genome.py",
        resumed_from={"trajectory": os.path.basename(traj_path),
                      "iterate": args.which,
                      "step": block.get("step"),
                      "loss": block.get("loss"),
                      "steps_completed": done,
                      "steps_remaining": remaining,
                      "warm_restart_only": (
                          "Adam moments, the secant delta0, n_reject and the decayed lr are "
                          "not recoverable from a trajectory; this restores position only.")},
        note=("Warm-restart start point recovered from a Stage-3 trajectory by "
              "studies/stage3_resume_genome.py. NOT a promotion candidate and not a "
              "descended genome: it is one iterate of an unfinished run."))

    recorded_hash = block.get("genome_hash")
    if recorded_hash is not None and recorded_hash != written_hash:
        raise SystemExit(
            f"round-trip check FAILED: the trajectory records {recorded_hash} for its "
            f"`{args.which}` iterate, the written record hashes to {written_hash}. The "
            f"adapter changed the genome; do not restart from {out}.")

    argv = restart_command(settings, out_path, remaining, stem + "_r2.json", lr0)

    print(f"wrote {out_path}  (genome_hash {written_hash}, "
          f"matches the trajectory's own {recorded_hash})")
    print(f"  from {args.which} iterate: step {block.get('step')} of {done} completed, "
          f"loss {block.get('loss')}")
    if remaining is not None:
        print(f"  {remaining} of {settings.get('steps')} steps remain")
    print("\nrestart with (WARM restart — Adam's moments start from zero, see the module "
          "docstring):\n")
    # argv[:3] is the interpreter and the script; everything after is flag/value pairs.
    pairs = argv[3:]
    lines = [" ".join(argv[:3])] + [" ".join(pairs[i:i + 2])
                                    for i in range(0, len(pairs), 2)]
    print("  " + " \\\n    ".join(lines))
    print("\n  The four flags after --lr are NOT recorded in the trajectory's `settings`; "
          "they are\n  this tree's defaults, and are wrong if the killed run set them by "
          "hand.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
