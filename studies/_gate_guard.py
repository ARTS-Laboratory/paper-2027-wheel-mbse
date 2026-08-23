"""Refuse to file a degraded run under a committed artifact's name.

PLAN.md §41, §43.  Every study driver's `--out` defaults to the artifact `make studies`
commits, and no fidelity flag changes that default.  So the cheap invocation — the one
someone reaches for while exploring — silently overwrites the gate's own record with a
weaker measurement, and the report still reads like the gate because every field is
present and every verdict is computed.

`study_contact` was the only driver that guarded this at all, and §41 measured what the
gap costs: at `--quick` its G1 reads 4.394e-04 with BOTH halves passing, against
1.7198e-03 and a red `regime_pass` at the real config.  Not a coarse gate standing in for
a fine one — a FALSE GREEN standing in for a RED one.

WHY A SHARED HELPER AND NOT NINE COPIES.  The MECHANISM is identical everywhere: compare
`--out` against the committed name, collect the reasons this run is not the gate, refuse
by name.  The JUDGEMENT is not — what degrades `study_mesh_quality` (fewer samples) has
nothing to do with what degrades `study_gradient` (a strain measure).  So the mechanism
lives here and every driver passes its own list.  This is `tests/test_fem.py:298-301`'s
argument applied to a guard instead of a check: one definition, so a fix to the wording
or the failure mode reaches all nine.

WHY REFUSE RATHER THAN REDIRECT.  `wheel_fea.py --smoke` retargets itself to
`best_solution_smoke.json`, which is the right call there because a smoke run is a normal
part of using the optimizer.  A degraded STUDY run is not routine, and `study_contact`'s
own guard argued the case when it was the only one: refuse "by name rather than by
silently renaming the output, because a `study_contact.json` holding three of seven
sections would read as the gate to everything downstream".  A silent rename also invents
filenames nobody asked for, next to artifacts that ARE tracked.

WHAT IS DELIBERATELY NOT GUARDED.  A different `--seed` on the two sampling drivers.  A
re-draw is still the full gate — same sample count, same config, same criterion — and
requiring seed 0 would pin a particular random draw rather than a fidelity.  `--samples`
ABOVE the default is likewise fine and only `--samples` below it is refused: a stronger
statistic is not a degraded one.
"""


def refuse_degraded_out(ap, args, committed, degraded):
    """Call `ap.error` if a degraded run is aimed at the committed artifact name.

    `degraded` is an iterable of `(is_degraded, reason)`.  Reasons are collected rather
    than short-circuited so the message names every problem at once — being told about
    `--quick`, fixing it, and then being told about `--config` is the kind of guard people
    route around.

    Checked against `args.out` by string equality, matching how the drivers already
    resolve it (`os.path.join(HERE, args.out)`), so any explicit path — even one that
    resolves to the same file — is accepted.  The guard is about not filing a weak run
    under the gate's name BY DEFAULT, not about protecting the inode from someone who
    typed the path deliberately.
    """
    if args.out != committed:
        return
    reasons = [r for is_degraded, r in degraded if is_degraded]
    if not reasons:
        return
    ap.error(
        f"this run is not the {committed} gate and may not be filed as it — "
        + "; ".join(reasons)
        + f".  Pass an explicit --out (e.g. --out {committed[:-5]}_probe.json)."
    )
