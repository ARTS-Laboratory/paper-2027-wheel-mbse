"""
M4 verification of the full-wheel linear FEA.

The invariants here are cheap and each one is an end-to-end check on the whole chain —
mesh, seams, constraints, quadrature, solve.  Rotational periodicity in particular
already earned its keep: it caught a formulation bug in which the contact phase moved
the patch around the rim instead of rolling the wheel under a fixed ground, which is a
different load case and was silently giving a 1.7% asymmetry.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import wheel_fea as wf            # noqa: E402
import wheel_fem as fem           # noqa: E402
import wheel_genome as wg         # noqa: E402
import wheel_wheel as ww          # noqa: E402
import study_wheel_fea as swf     # noqa: E402
import study_reds_ratio_stability as RS   # noqa: E402  — the retired `max/min` gate's
#                                         # replacement constants and the grid behind them

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = "coarse"


@pytest.fixture(scope="module")
def genes():
    with open(os.path.join(REPO, "best_solution.json")) as fh:
        return wg.genes_to_vector(json.load(fh)["genes"])


@pytest.fixture(scope="module")
def mesh(genes):
    return ww.build_wheel(genes, CFG)


@pytest.fixture(scope="module")
def res(mesh):
    return fem.solve_wheel(mesh)


# A SECOND MESH, AND DELIBERATELY NOT A CHANGE TO THE ONE ABOVE (PLAN.md §109).
#
# `fillet=True` is what `wheel_objective.phase_meshes` and `wheel_pool_worker.run_phase`
# pass unconditionally, so this is the wheel the optimizer actually solves.  Exactly one
# test below reads it, and the temptation is to give `mesh` the flag instead and let the
# whole file follow -- which would silently re-aim the ELEVEN other tests that read them at
# a mesh none of them was calibrated on.  MEASURED RATHER THAN ASSERTED (§109, from the two ladders in
# `studies/study_reds_hub_share.json` at the shipped genome, `coarse`):
#
#   `0.25 < rim   < 0.40`   0.3113 -> 0.3781   holds, but eats 44.6% of the band
#   `0.58 < spoke < 0.72`   0.6545 -> 0.6136   holds, but eats 29.3% of the band
#   `1.4  < drop  < 2.0`    1.5516 -> 0.9614   BREAKS, and at every rung on the ladder
#
# So a shared fixture would not have re-aimed six gates quietly; it would have turned one
# of them red on the spot and moved two others most of the way to their edges.  Moving one
# gate is the judgement §109 made.  Moving twelve is not, and one flag on `mesh` would have
# made it look like one edit.
#
# Module-scoped and therefore lazy: costs a build and a ~1.1 s solve only on the runs that
# reach the test that asks for it.
@pytest.fixture(scope="module")
def filleted_mesh(genes):
    return ww.build_wheel(genes, CFG, fillet=True)


@pytest.fixture(scope="module")
def filleted_res(filleted_mesh):
    return fem.solve_wheel(filleted_mesh)


# ---------------------------------------------------------------------------
# EQUILIBRIUM AND THE LOAD
# ---------------------------------------------------------------------------

def test_hub_reaction_balances_the_applied_load(res):
    """Independent check on the entire assembly, and nearly free since K*u is formed."""
    assert res["equilibrium_error_n"] < 1e-6, res["equilibrium_error_n"]
    assert abs(res["applied_force_n"][1] - wf.TOTAL_FORCE_NEWTONS) < 1e-9


def test_the_ground_traction_is_vertical(mesh):
    """A rigid FLAT frictionless ground pushes along ITS normal, not the rim's.

    Using the rim's radial normal instead leaves a horizontal resultant that the hub has
    to react — a side load the real problem does not have.  At a 12 degree patch that
    error is 2%, and it silently breaks the rotational periodicity below.
    """
    _, f = fem.wheel_problem(mesh)
    fx = float(f.reshape(-1, 2)[:, 0].sum())
    assert abs(fx) < 1e-9, f"horizontal resultant {fx:.3e} N"


def test_only_the_rim_od_near_the_bottom_is_loaded(mesh):
    """The load must sit on the rim OD, centred at the bottom, and nowhere else.

    The angular bound allows ONE element beyond the patch edge, and that is correct
    rather than sloppy: consistent nodal loads come from integrating the traction against
    shape functions, so a node just outside the patch still picks up load from Gauss
    points just inside it.  Requiring the loaded nodes to lie strictly within the patch
    would be demanding lumped loads, which is the error the M3 traction patch test
    exists to catch.

    "ONE ELEMENT" USED TO BE `SECTOR_DEG / (n_weld + n_rim_free)`, AND THAT IS A MEAN, NOT
    AN ELEMENT.  The rim OD's two families are not the same size — measured at `coarse`,
    10 weld segments of 0.1682 deg plus 10 free-arc segments of 2.8318 deg make the sector
    up to 30.000 deg exactly, a span ratio of 16.8 that holds at every config.  So the old
    expression returned 1.5 deg, the average of two sizes an order of magnitude apart,
    while the element that actually straddles the patch edge is the free-arc one at
    2.8318 deg.  The bound understated a real element by 1.888x and the test passed only
    while the patch edge happened to fall on a favourable part of the node grid; it went
    red on the 2026-08-13 promotion when it stopped doing so.

    PLAN.md section 19 diagnosed the 1.9x as quadratic elements spanning two node pitches.
    That is not the cause — the element ORDER has nothing to do with it, and a factor of 2
    would be the wrong fix for the same reason the mean was: it is another constant
    standing in for a number the mesh already knows.  The bound is now read off the mesh.
    """
    half = 3.0
    _, f = fem.wheel_problem(mesh, patch_half_deg=half)
    xy = np.asarray(mesh.coords)
    loaded = np.linalg.norm(f.reshape(-1, 2), axis=1) > 0
    assert loaded.sum() > 0
    th = np.degrees(np.arctan2(xy[loaded, 1], xy[loaded, 0])) % 360.0

    # The widest rim-OD element there actually is, measured, wrap-safe.
    seg = np.asarray(mesh.edge_sets["rim_outer"])
    ends = np.degrees(np.arctan2(xy[seg[:, (0, 2)], 1], xy[seg[:, (0, 2)], 0]))
    element_deg = float(np.abs((np.diff(ends, axis=1) + 180.0) % 360.0 - 180.0).max())

    assert np.abs(((th - 270.0 + 180.0) % 360.0) - 180.0).max() <= half + element_deg
    assert np.abs(np.linalg.norm(xy[loaded], axis=1)
                  - ww.RIM_OUTER_RADIUS_MM).max() < 1e-9


def test_work_identity(res):
    """delta = 2U/F for a linear body under one load system.

    Ties the reported strain energy to the reported displacement, so the compliance
    split — which is computed from energies — cannot drift away from the axle drop it
    claims to decompose.  The work-conjugate displacement is the pressure-weighted mean
    over the patch, which sits between the centre node and the plain patch mean, so it
    is checked to bracket rather than to equal either.
    """
    delta_work = 2.0 * res["strain_energy_mJ"] / wf.TOTAL_FORCE_NEWTONS
    lo = min(res["axle_drop_mm"], res["axle_drop_patch_mean_mm"])
    hi = max(res["axle_drop_mm"], res["axle_drop_patch_mean_mm"])
    assert lo - 1e-9 <= delta_work <= hi + 1e-9, (
        f"2U/F = {delta_work:.6f} outside [{lo:.6f}, {hi:.6f}]")


@pytest.mark.parametrize("phase", [0.0, 11.0])
def test_axle_drop_is_exactly_12_fold_periodic(genes, phase):
    """delta(phi) = delta(phi+30) to solver precision.

    The strongest cheap end-to-end check available: it exercises the mesh, the seams,
    the sector indexing, the load, and the solve, and it has an exact expected answer.
    The wheel is CHIRAL — twelve spokes all spiralling the same way — so the plan's
    mirror-symmetry check does not exist for this geometry and this replaces it.
    """
    a = fem.solve_wheel(ww.build_wheel(genes, CFG, phase_deg=phase))["axle_drop_mm"]
    b = fem.solve_wheel(ww.build_wheel(genes, CFG,
                                       phase_deg=phase + 30.0))["axle_drop_mm"]
    assert abs(a / b - 1.0) < 1e-9, f"{a:.10f} vs {b:.10f}"


def test_the_wheel_has_no_mirror_symmetry(mesh):
    """Pin the chirality, because it is why the plan's mirror check was dropped.

    If someone later makes the spokes straight or symmetric, the mirror check becomes
    available and should be reinstated — this test failing is that signal.
    """
    from scipy.spatial import cKDTree
    xy = np.asarray(mesh.coords)
    mirrored = xy * np.array([1.0, -1.0])
    d, _ = cKDTree(xy).query(mirrored)
    assert d.max() > 0.1, (
        f"the wheel is mirror-symmetric to {d.max():.3e} mm — the chirality argument "
        f"no longer holds and the mirror-symmetry check should be reinstated")


# ---------------------------------------------------------------------------
# THE GATE NUMBERS
# ---------------------------------------------------------------------------

def test_compliance_split_is_a_partition(res):
    s = res["compliance_split"]
    assert abs(sum(s.values()) - 1.0) < 1e-12
    assert all(v >= 0.0 for v in s.values())


def test_the_rim_band_holds_a_large_minority_of_the_compliance(res):
    """The 1.5 mm rim band holds about a third of the compliance.

    It was 44.7% with the 1.1 mm band that shipped before M4; thickening the band inward
    to 1.5 mm and re-running the GA brought it to ~32%.  Pinned as a range because it
    moves slightly with mesh and patch, and deliberately WIDE at the bottom: what must
    not change silently is that the rim is a first-order term the beam model omits
    entirely, not that it holds any particular share.

    THE HUB THIRD OF THIS ASSERTION MOVED OUT, to the xfail below.  It is a live question
    about the wheel rather than a fact about the rim band, it has been red since §14, and
    leaving it here meant the rim and spoke shares — which pass, and which are what this
    test is named for — were not being checked at all on any run.  See PLAN.md §31.
    """
    s = res["compliance_split"]
    assert 0.25 < s["rim"] < 0.40, s
    assert 0.58 < s["spoke"] < 0.72, s


def test_the_hub_junction_holds_a_small_minority_of_the_compliance(filleted_res):
    """RED FROM §14 TO §108, AND GREEN HERE BECAUSE THE INSTRUMENT CHANGED, NOT THE WHEEL.

    THE NAME NO LONGER CARRIES THE NUMBER, ON PURPOSE.  This was
    `..._holds_under_three_percent_of_the_compliance` and the three percent is gone.  A
    name that quotes its own threshold goes stale silently the day the threshold moves,
    and there is nothing to make it fail -- the assertion below still passes while the name
    above it lies.  The bound lives in the assertion now.  (The assertion has moved once
    before, out of `..._rim_band_holds_a_large_minority_...` at §31, which is why that
    older name is the one most of PLAN.md's early sections use for this gate.)

    WHAT §31 DECIDED, AND WHY IT NO LONGER HOLDS.  §31 (2026-08-15) put the question up
    with numbers, had it handed back, and made the call itself: *the bound stays at 0.03
    and this stays red, as an accepted deficit.*  That call rested on exactly two legs,
    both stated in §31 and both now gone:

      (A) `0.03` IS ACHIEVABLE -- `best_solution_ga_beam.json`, "the design the 3% bound
          was calibrated on", met it converged with 53% to spare.
      (B) `0.03` IS NOT A MESH ARTEFACT -- the shipped genome was over at the COARSEST
          rung, before any refinement argument starts.

    (B) IS FALSIFIED.  §106 re-ran the ladder on the filleted mesh and the level fell by
    76%; the ladder is on disk here as of §109, both arms, five rungs, linear on both
    sides so the only difference between them is the mesh
    (`studies/study_reds_hub_share.json`, `rungs` and `rungs_filleted_linear`;
    `make reds-hub` and `make reds-hub-fillet-rungs`):

        genome    mesh       smoke     coarse    medium    fine      ultra     drift
        shipped   plain      0.032489  0.034188  0.035237  0.036483  0.037053  +14.05%
        shipped   FILLETED   0.008079  0.008308  0.008409  0.008442  0.008463   +4.74%
        ga_beam   plain      0.013823  0.013722  0.013849  0.014043  0.014144   +2.32%
        ga_beam   FILLETED   0.005312  0.005381  0.005406  0.005417  0.005423   +2.09%

    It WAS the mesh.  The plain increments are +0.001699, +0.001049, +0.001246, +0.000570
    -- not monotone, rising at `medium`->`fine`, which is why this quantity could never be
    called converged.  The filleted ones are +0.000228, +0.000101, +0.000033, +0.000020,
    falling at every rung.  For the filleted ladder to climb to 0.03 its increments would
    have to stop decaying almost exactly -- a ratio of 1.00095 against the 1.640 measured
    at the top of the ladder -- so the pass here is not resting on the extrapolation.

    (A) IS GONE, AND MORE COMPLETELY THAN "THE NUMBER MOVED".  `ga_beam` asks for
    `R_hub` 1.5598 / `R_rim` 3.0, and on the filleted mesh THAT WHEEL DOES NOT EXIST:
    an explicit `fillet=(1.5598, 3.0)` is refused outright -- *"the fillet's tangent point
    has passed the next sector's corner (-8.400 deg of free ring left)"*.  The filleted
    `ga_beam` row above is `fillet=True`, which keeps the genome and CLAMPS it to
    (0.667, 0.895) (`wheel_wheel.SECTOR_FIT_CLAMP`).  So the design the bound was
    calibrated on cannot be built on the mesh the objective solves, and leg (A) has no
    referent there at all.

    WHY THE BOUND COULD NOT SIMPLY BE CARRIED OVER, WHICH IS THE TRAP THIS TEST WAS ONE
    EDIT AWAY FROM.  Pointing the fixture at the filleted mesh and leaving `0.03` alone
    passes -- by 72% -- and it takes the calibration design's margin from 54.3% to 82.1%
    at this rung.  That is §14's own prohibition running backwards: §14 forbids moving a
    bound to admit the design that breached it, and holding a bound still while moving the
    instrument under it has the same effect by the other route.  A gate sitting 3.6x above
    the quantity it watches is not a gate.

    AND IT COULD NOT BE RESCALED BY A FACTOR EITHER, WHICH IS THE LESS OBVIOUS HALF.  The
    mesh does not rescale this quantity -- it rescales it DIFFERENTLY PER DESIGN.  At
    `coarse` the shipped genome falls 4.115x and `ga_beam` falls 2.550x, because the
    unfilleted re-entrant corner penalises a thin hub junction far harder than a thick
    one, which is the same mechanism §30 measured on the rim corner.  There is no single
    transport factor for `0.03` to ride across.

    THE CALL, MADE IN §109 (2026-09-04): THE MESH MOVES TO THE FILLETED BUILD, THE BOUND
    MOVES TO `0.0117`, AND THIS STOPS BEING AN XFAIL.

    `0.0117` PRESERVES §31's WARRANT RATHER THAN §31's NUMBER.  What made `0.03`
    defensible was leg (A): the reference design cleared it by 54.3% at this rung.  So the
    bound is rescaled by the REFERENCE design's own mesh factor and never by the design
    under test -- calibrating on the genome being gated is the §14 sin itself:

        0.03 * (0.005381416728939758 / 0.013722004451848286) = 0.011765...

    rounded DOWN to 0.0117 so the gate is never looser than its derivation.  `ga_beam`
    keeps 54.0% of margin, against the 54.3% it had on the plain mesh at `0.03`; the
    shipped genome clears it by 29.0%.

    THE ONE SCOPE LIMIT, AND ITS DIRECTION IS KNOWN.  The factor above comes from the
    CLAMPED `ga_beam`, because the unclamped one does not build -- so it is a bound on a
    stand-in, and the stand-in's hub fillet (0.667 mm) happens to land within 0.51% of the
    shipped genome's (0.664 mm) while its rim fillet (0.895 mm) does not (3.0 mm).  §14's
    direction -- the hub share RISES as `R_hub` FALLS, which §75 confirmed on a mesh that
    can express it -- says an unclamped `ga_beam` would read LOWER, so the true factor is
    SMALLER and the honest bound is TIGHTER than 0.0117.  `0.0117` is the loose end of the
    range, and the shipped genome's 29.0% is therefore an upper bound on its own margin.

    WHAT IS NOT CLAIMED HERE.  Not that the wheel improved: `best_solution.json` is
    untouched and its hub junction is what it always was.  What changed is that the mesh
    stopped putting a singularity where the fillet is.  §31's design successor -- `cy4`
    alone moves the plain-mesh share by 102% of the gap, so hub compliance is reachable
    from Stage 2/3's objective if it is ever worth constraining -- is not retired by this,
    it is de-prioritised: HUBSHARE Step 0 asked whether such a term would buy anything and
    the answer on this mesh is no.

    THE SIBLING GATE ABOVE STILL READS THE PLAIN MESH.  `..._rim_band_holds_a_large_
    minority_...` and the ten other tests on the `mesh`/`res` fixtures were calibrated there
    and stay there; see the `filleted_mesh` fixture for the measured reason that is
    deliberate rather than lazy.
    """
    assert filleted_res["compliance_split"]["hub"] < 0.0117, (
        filleted_res["compliance_split"])


def test_the_beam_model_does_not_predict_the_axle_drop(res, genes):
    """THE M4 HEADLINE: the beam model's 2.0 mm target is not what the part does.

    The sign of this has flipped once already — the wheel was 42.7% SOFTER than the
    target with the 1.1 mm band and is now stiffer than it — so the test pins the
    magnitude of the disagreement rather than its direction.  A genuine improvement (a
    re-tuned band, a Stage-2 objective that actually sees the wheel) should show up here
    as a failure to be looked at rather than passing unnoticed.
    """
    beam = wf.evaluate_design(genes)[0]["deflection_mm"]
    ratio = res["axle_drop_mm"] / beam
    assert abs(ratio - 1.0) > 0.10, (
        f"axle drop {res['axle_drop_mm']:.4f} mm vs beam {beam:.4f} mm — the beam model "
        f"has become predictive on this genome, which would be news; check whether "
        f"study_wheel_fea.run_beam_blindness still finds a spread across genomes before "
        f"believing it")
    assert 1.4 < res["axle_drop_mm"] < 2.0, res["axle_drop_mm"]


def test_the_beam_to_wheel_ratio_is_not_a_constant(genes):
    """Gate 1's conclusion: the plan's Stage-2.5 off-ramp does not exist.

    The off-ramp was "correct the beam model with one factor and skip Stages 2 and 3".
    That factor is `axle_drop / beam_deflection`, so it exists only if that ratio is
    roughly constant over the design space.  It is nowhere near constant: its coefficient
    of variation is 0.1450 at worst over the 66 beam cells measured across both gene
    boxes, against the 0.10 bar the off-ramp would need.

    Reduced fidelity (smoke mesh, few samples per seed) on purpose — the dispersion is
    large and does not need a converged mesh to be visible.  If this ever passes, the
    whole Stage 2 justification needs re-reading, which is why it fails loudly.

    IT PINS THE FLOOR AT 2.0 BECAUSE THE STATISTIC IS A PROPERTY OF THE GENE BOX, NOT OF
    `genes`.  `run_beam_blindness` draws a Latin hypercube from the box and computes its
    statistics over the DRAWN rows, explicitly excluding the genome it is passed — so they
    do not depend on which design ships, and they do depend on where the box's thickness
    floor sits.  Measured (§14):

        genome        floor 2.0    floor 1.2
        36aed36           4.943        2.686
        350f4c7           4.943        2.686

    Identical down the genome column, to every digit.  §13's move of the DEFAULT floor to
    1.2 therefore moved this number without touching the wheel: a lower floor admits
    floppier random spokes.  The floor stays pinned at 2.0 here — see the note at the
    bottom of this docstring for why that is now a loose end rather than a reason.

    ===========================================================================
    `fea_over_beam_ratio > 3.0` WAS RETIRED IN THE REDS ARC.  DO NOT REINTRODUCE IT.
    ===========================================================================
    It was a `max/min` over the drawn rows, which is an estimator of the sample RANGE and
    therefore grows without bound with the number of draws.  It was never a property of
    the design space, and a gate cannot be placed on it.  Measured over 109 cells by
    `studies/study_reds_ratio_stability.py` (its module comment carries the full table):

        20 seeds at this test's own n=6 :  ratio 1.570 - 30.129, passing `> 3.0` in 7/20
        n = 6, 12, 24, 48, 96 at seed 7  :  2.413, 9.995, 34.968, 34.968, 48.123

    SEED 7 — the seed this test hard-coded — IS THE LOW OUTLIER of the twenty.

    AND THE NUMBER MOVED WITHOUT THE WHEEL MOVING, which is the same lesson twice.  §14
    measured 4.943 at this very floor and seed, where it now reads 2.413 — and this
    statistic explicitly EXCLUDES the shipped genome, so no promotion can account for it.
    A property of the gene box moves when the BOX moves, and the box did: `R_hub`'s floor
    went 0.5 -> 0.4 on 2026-08-11 (`wheel_fea.py:282`, BUILD_PLAN steps 5 and 6), after
    §14's figure was taken.  Not chased further, because it does not need to be — a
    quantity that a barrier-bound edit can halve is not one to hang a threshold on, which
    is the conclusion either way.

    What replaces it is a bound on the CV, which is the arithmetic
    `correction_factor_is_defensible` is DEFINED in terms of (`cv < 0.10`, study_wheel_fea
    line ~400) — so this is PLAN §28's move, a stale constant replaced by the claim's own
    arithmetic, not a loosened bound.  `cv > 0.14` strictly implies the first assertion
    below.  The bound is derived, not picked: 0.14 is the CV's measured floor over all 109
    cells (0.1450), floored to two decimals, and the same constant serves both this test
    and its GNL twin so neither is tuned to its own run.

    AND IT IS CHECKED AT FIVE SEEDS, NOT ONE, which is the specific defect that let a
    seed lottery sit here for four arcs.  The old line asked one draw; this asks the
    ensemble's worst.

    LOOSE END FOR A HUMAN, NOT ACTED ON HERE: the only stated reason this test pins the
    wall floor at 2.0 was that the `> 3.0` margin had been calibrated in a 2.0 mm box.
    That margin is now retired, so the pin has no rationale left.  The REDS arc measured
    the replacement in both boxes rather than move it — the 1.2 mm box's CV floor is
    0.1948 over the same 20 seeds, comfortably above the same 0.14 gate — so dropping the
    pin would not change any verdict.  §14 called re-deriving Gate 1 at the 1.2 floor "a
    real piece of work and a judgement about Gate 1"; the measurement is now done and the
    judgement is still a human's.  See PLAN.md §31.
    """
    before = wf.MIN_WALL_MM
    try:
        wf.set_min_wall(2.0)
        reps = {s: swf.run_beam_blindness(genes, "smoke", n=6, seed=s)
                for s in RS.RETIREMENT_SEEDS}
    finally:
        # Restore unconditionally.  `tests/test_stage3.py` takes its bounds in a
        # MODULE-scoped fixture that never recomputes, so a floor leaked from here would
        # not merely persist — it would be baked in and fail somewhere else entirely.
        wf.set_min_wall(before)

    assert not any(r["correction_factor_is_defensible"] for r in reps.values()), {
        s: r["fea_over_beam_cv"] for s, r in reps.items()}
    worst = min(reps.items(), key=lambda kv: kv[1]["fea_over_beam_cv"])
    assert worst[1]["fea_over_beam_cv"] > RS.GATE_CORRECTION_CV, (
        f"beam-to-wheel correction CV fell to {worst[1]['fea_over_beam_cv']:.4f} at seed "
        f"{worst[0]}, under the {RS.GATE_CORRECTION_CV} gate — the correction is becoming "
        f"uniform and the Stage-2.5 off-ramp is reopening at its own 0.10 bar.  That is "
        f"news, not a gate to move: re-run studies/study_reds_ratio_stability.py and read "
        f"PLAN.md §31 before touching this number")


def test_the_retired_max_min_gate_is_decided_by_the_sample_size(genes):
    """Keep the REASON `fea_over_beam_ratio > 3.0` was retired measured, not just asserted.

    A docstring saying "max/min is sample-size dependent" is an argument.  This is the
    demonstration, and it exists so that the next person to look at the report dict — which
    still publishes `fea_over_beam_ratio`, deliberately, because it is a useful diagnostic
    — cannot mistake it for something a threshold could sit on.

    It asserts the retired gate's VERDICT FLIPS with `n` at a fixed seed: the same wheel,
    the same box, the same seed, one number below 3.0 and one above.  Measured at seed 7,
    the seed the retired test hard-coded: 2.413 at n=6 and 34.968 at n=24.

    If this ever fails, the ratio has become sample-size stable and the retirement argument
    needs re-reading — which is the point of pinning it rather than deleting it.
    """
    before = wf.MIN_WALL_MM
    try:
        wf.set_min_wall(2.0)
        small = swf.run_beam_blindness(genes, "smoke", n=6, seed=7)["fea_over_beam_ratio"]
        large = swf.run_beam_blindness(genes, "smoke", n=24, seed=7)["fea_over_beam_ratio"]
    finally:
        wf.set_min_wall(before)

    assert small < 3.0 < large, (
        f"max/min over the drawn rows read {small:.3f} at n=6 and {large:.3f} at n=24 — "
        f"it no longer brackets the retired 3.0 gate, so the demonstration that the gate's "
        f"verdict was decided by the sample size has stopped working")


def test_the_free_arc_fraction_is_not_constant_over_the_design_space(genes):
    """The variable that explains gate 1 must actually vary, or the explanation is empty.

    `spoke_free_arc_fraction` is the share of the spoke that is not swallowed by the two
    weld arcs.  It is what a Stage-2 objective would use, and a term on a near-constant
    would be worthless — so this pins that it moves, and that the shipped genome is not
    at an extreme of it.
    """
    rep = swf.run_beam_blindness(genes, "smoke", n=6, seed=7)
    fa = [r["free_arc_fraction"] for r in rep["rows"]]
    assert max(fa) - min(fa) > 0.05, fa
    assert all(0.0 < f < 1.0 for f in fa), fa


def test_stiffening_only_the_rim_helps_more_than_its_energy_share(mesh, res):
    """The load-spreading effect: rigidifying the rim removes far more than 44.7%.

    Both numbers are true and they answer different questions.  This pins the ORDER —
    if the rigid-rim figure ever drops below the energy share, one of them is wrong.
    """
    rigid = fem.solve_wheel(mesh, rim_modulus_scale=1000.0)
    removed = 1.0 - rigid["axle_drop_mm"] / res["axle_drop_mm"]
    assert removed > res["compliance_split"]["rim"], (
        f"rigid rim removed {removed:.2%} but the energy share is "
        f"{res['compliance_split']['rim']:.2%}")
    assert removed > 0.6


def test_a_thicker_rim_monotonically_stiffens_the_wheel(genes):
    """The sign of the rim's effect, which is what this test is named for.

    Monotonicity passes at every rung in the tree and always has — REDS measured it at
    smoke, coarse and medium.  The absolute assertion that used to sit under it did not,
    and it was retired rather than moved.  The full measurement (drops in mm, for
    rim_outer 49.7, 50.0, 50.6, 51.2):

        rung     49.7    50.0    50.6    51.2    monotone?   brackets 2.0?   sweep cost
        smoke  1.8758  1.5453  1.1823  0.9813      yes            NO           0.7 s
        coarse 1.9798  1.6208  1.2320  1.0193      yes            NO           1.9 s
        medium 2.0034  1.6399  1.2462  1.0308      yes            yes          4.7 s

    ===========================================================================
    `drops[-1] < TARGET_DEFLECTION_MM < drops[0]` WAS RETIRED IN THE REDS ARC.
    ===========================================================================
    It is an ABSOLUTE deflection claim, and it was being evaluated at `smoke`, the least
    converged mesh in the tree — which reads about 6% low (PLAN §29's ladder: -5.955%
    under SVK).  That is more than enough to lose a bracket whose upper edge only reaches
    2.0 mm by `medium`.  PLAN §29 retired exactly this class of claim at the plan level:
    an absolute distance from 2.0 mm quoted without naming its rung.

    MOVING IT TO `medium` WAS MEASURED AND REJECTED, and the reason is not cost — the
    medium sweep is only +4.0 s.  It is that `medium` cannot support the claim either:

        rung     drop at 49.7    margin over 2.0    drift from previous rung
        medium       2.0034          +0.169%
        fine         2.0134          +0.672%              +0.50%

    The margin at `medium` (0.17%) is SMALLER THAN THE QUANTITY'S OWN REMAINING
    DISCRETISATION DRIFT (0.50% from medium to fine).  A bound a number clears by less
    than its own convergence error is not a gate, it is a coin toss that happens to be
    landing the right way — which is the §29 lesson one rung up.

    SO THE FINDING IS RECORDED HERE RATHER THAN ASSERTED: the target IS bracketed, at
    `medium` (2.0034 > 2.0 > 1.0308) and more comfortably at `fine` (2.0134).  It is not
    bracketed at `smoke` or `coarse`, and nothing in this tree should quote it without a
    rung attached.

    What replaces it is the SPAN, which is a ratio and therefore survives the mesh: the
    thinnest rim is ~1.9x softer than the thickest at every rung (1.912 smoke, 1.942
    coarse, 1.944 medium — 1.7% total drift, against 6% on the absolute drops).  Gated at
    1.5, which the measured floor of 1.912 clears by 27%.  That keeps a magnitude on the
    effect without making an absolute claim at a rung that cannot carry one.
    """
    drops = [fem.solve_wheel(ww.build_wheel(genes, "smoke", rim_outer=ro))["axle_drop_mm"]
             for ro in (49.7, 50.0, 50.6, 51.2)]
    assert all(drops[i + 1] < drops[i] for i in range(len(drops) - 1)), drops
    span = drops[0] / drops[-1]
    assert span > 1.5, (
        f"a 1.5 mm rim_outer sweep moved the axle drop by only {span:.3f}x {drops} — the "
        f"rim's first-order effect has collapsed, which contradicts the compliance split "
        f"above; measured 1.912 (smoke) / 1.942 (coarse) / 1.944 (medium) in REDS")


# ---------------------------------------------------------------------------
# WHAT IS AND IS NOT A CONVERGED NUMBER
# ---------------------------------------------------------------------------

def test_peak_stress_diverges_but_the_field_converges(genes):
    """The unfilleted junction is a re-entrant corner, so the pointwise maximum must NOT
    be quoted as a stress — it grows without bound under refinement — while the p99 of
    the same field settles.  Asserting both directions keeps anyone from reading the max
    as a real number, and keeps the p99 from being quietly replaced by the max.

    THIS USED TO ASSERT `d2 < 0.3 * d1` ON THE p99's SUCCESSIVE DIFFERENCES, and that is
    a divergence detector that only means anything while `d1` is still a real
    discretization error.  Once the quantity has actually converged, `d1` and `d2` are
    both tail, their ratio is arbitrary, and the test fires on a wheel that is behaving
    perfectly.  The old comment below already named this failure mode one tier down —
    "the smoke and coarse values happen to sit close together, which makes the FIRST
    difference small and the second one look like divergence" — and §14 measured it
    happening one tier UP on the promoted genome, which converges sooner:

        genome        smoke    coarse    medium      fine     d2/d1    d2/p99
        350f4c7      18.327    17.274    17.246    17.230     0.573    0.094%
        36aed36       8.842     8.782     8.612     8.605     0.042    0.082%

    `d2/d1` = 0.573 fails the old gate; `d2/p99` = 0.094% says the p99 has settled to
    under a tenth of a percent.  The second number is the one that means "converged".
    And the old genome is not a counter-example — it FAILS the same ratio test at 2.816
    if the window starts at `smoke`.  The window had to be hand-picked per design, which
    is the tell that the statistic was wrong rather than the meshes.

    So this now pins the CONTRAST the docstring is actually about — one quantity running
    away while the other stands still — as a ratio of RELATIVE drifts, which is
    dimensionless and does not care which tier a given design converges on.
    """
    maxima, plain = [], []
    for cfg in ("coarse", "medium", "fine"):
        m = ww.build_wheel(genes, cfg)
        st = swf.stress_report(m, fem.solve_wheel(m))
        maxima.append(st["rim"]["max_singular_mpa"])
        plain.append(st["spoke_block_p99_mpa"])

    assert maxima[1] > maxima[0] and maxima[2] > maxima[1], (
        f"the corner singularity has stopped growing: {maxima} — either fillets were "
        f"added (good, update this test) or the stress recovery changed")
    # Monotone is not enough on its own: three noisy samples can be monotone by luck.
    # Measured 38.9% (350f4c7) and 34.7% (36aed36) over coarse..fine.
    max_drift = maxima[2] / maxima[0] - 1.0
    assert max_drift > 0.20, (
        f"the singular max grew only {max_drift:.1%} over coarse..fine {maxima} — that "
        f"is not a divergence, and the whole 'do not quote the max' argument rests on it")

    # The converging quantity has to be measured AWAY from the singular corner.  A
    # percentile over a region that contains the corner is not converged either, because
    # the number of near-corner Gauss points grows with refinement — which is why this
    # uses the plain spoke block and not the rim region's p99.
    p99_drift = abs(plain[2] - plain[0]) / plain[2]
    assert p99_drift * 10.0 < max_drift, (
        f"plain-spoke p99 drifted {p99_drift:.2%} over coarse..fine {plain} against the "
        f"max's {max_drift:.1%} — less than the 10x separation that makes one of these a "
        f"converged number and the other a mesh artifact")
    d2 = abs(plain[2] - plain[1])
    assert d2 / plain[2] < 0.01, f"plain-spoke p99 still moving {d2 / plain[2]:.2%}"


def test_the_junction_is_re_entrant_enough_to_be_singular(genes):
    """Tie the singularity to the geometry that causes it, in one number.

    THIS USED TO BE `test_the_arrival_angle_makes_the_junction_a_near_crack` AND TO
    ASSERT `material_wedge > 340.0`, which pinned a PATHOLOGY as an invariant — the same
    mistake as the `fillet_families == 2` assertion §13 replaced.  A design whose spokes
    arrive less tangentially has a LESS crack-like junction, which is an improvement, and
    it broke the test.  Measured: 349.5 deg on the GA/beam genome, 315.4 on the promoted
    one.

    What actually has to hold — and what `test_peak_stress_diverges_but_the_field_
    converges` above depends on — is only that the junction is re-entrant, so a stress
    singularity exists at all and the pointwise max is not a number.  That bound is not a
    property of one design: `MAX_ARRIVAL_DEG` caps the arrival angle for EVERY genome the
    optimizer can reach, so the wedge is at least `360 - MAX_ARRIVAL_DEG` = 295 deg
    across the whole box.  Asserting the derived bound rather than a measured value is
    what stops this from having to be re-fitted the next time a genome ships.
    """
    arrival = max(float(a) for a in ww.arrival_angles(genes, ww.get_config("coarse")))
    material_wedge = 360.0 - arrival

    assert arrival <= ww.MAX_ARRIVAL_DEG, (
        f"arrival {arrival:.1f} deg exceeds MAX_ARRIVAL_DEG {ww.MAX_ARRIVAL_DEG} — the "
        f"barrier that is supposed to enforce this let a genome through")
    assert material_wedge >= 360.0 - ww.MAX_ARRIVAL_DEG, (
        f"material wedge {material_wedge:.1f} deg is below the {360 - ww.MAX_ARRIVAL_DEG} "
        f"deg the arrival cap guarantees — the two are inconsistent, so one of them moved")
    assert material_wedge > 180.0, (
        f"material wedge {material_wedge:.1f} deg is no longer re-entrant, so the "
        f"junction is not singular and the convergence-rate explanation in "
        f"study_wheel_fea.py needs revisiting")


def test_compliance_split_is_robust_to_the_patch_assumption(mesh):
    """The axle drop depends on the assumed patch; the DECISION must not.

    Over a 12x range of patch size the rim share moves by ~7 points while the axle drop
    moves 13%, and the ordering never changes.  That is what licenses quoting the split
    before M6 replaces the assumed patch with real contact.
    """
    shares = [fem.solve_wheel(mesh, patch_half_deg=h)["compliance_split"]["rim"]
              for h in (1.0, 3.0, 12.0)]
    assert all(0.22 < s < 0.38 for s in shares), shares
    assert shares[0] > shares[-1], shares


# ---------------------------------------------------------------------------
# MASS
# ---------------------------------------------------------------------------

def test_total_mass_matches_the_step_manifest_within_the_embed_difference(mesh):
    """Resolves the old "two mass figures are not comparable" wart.

    `metrics.total_mass_g` is spokes only; `wheel_mass_g` is the whole solid.  It lands
    under the manifest's `mass_g_pla` by TWO differences: the `_embed` gusset, and the
    fillet material, which the mesh does not model at all.

    THIS USED TO ASSERT A PERCENTAGE BAND — `-0.030 < m/manifest - 1 < -0.012` — AND THE
    MANIFEST PUBLISHED NEITHER OF THE TWO TERMS THAT BAND WAS STANDING IN FOR.  The
    docstring decomposed the gap into "~1.4% gusset plus 0.92% fillets" and nothing
    computed either number; both were fitted to one wheel.  On the promoted genome the
    band failed at -6.9% and there was no way to tell which term had moved, because there
    was nothing to look at.

    §14 made the fillet term measurable instead of guessed.  `wheel_step_export` builds
    `wheel_nofillet.step` anyway as its fallback, so publishing that solid's volume turns
    the fillet material into a subtraction OCC does exactly:
    `fillets.volume_mm3 = solid.volume_mm3 - solid.volume_nofillet_mm3`, measured before
    despecialization on both sides.  It is 2372.53 mm^3 — **6.18% of the solid**, not the
    0.92% the old docstring claimed.  That one correction is most of the -6.9%.

    So the budget is now checked as a budget.  Subtract the fillets, which are a published
    number, and what is left over must be the gusset alone: 0.70% of the solid, positive,
    and small.  A gap that is neither is a real modelling difference and should be read
    rather than absorbed into a wider band.

    WHAT THIS DELIBERATELY DOES NOT USE IS `EMBED_ALLOWANCE_PER_SPOKE_MM2`.  That constant
    is 3.03 and §14 measured the promoted genome's actual gusset at **0.98 mm^2 per
    spoke** — it is stale, in the same way `MIN_JUNCTION_OVERLAP_MM3` was stale in §12,
    and for the same reason: a fixed mm^2 standing in for something that scales with root
    thickness.  Left as an open item, because guessing a new constant would only re-stale
    it on the next genome.  Nothing here depends on it.
    """
    with open(os.path.join(REPO, "export", "wheel_step_manifest.json")) as fh:
        man = json.load(fh)
    solid, fil = man["solid"], man["fillets"]

    # (1) The new field is what it claims to be.  Guards a sign or unit slip in the
    #     exporter, which the budget below would otherwise silently absorb.
    assert fil["volume_mm3"] == pytest.approx(
        solid["volume_mm3"] - solid["volume_nofillet_mm3"], abs=0.2), (
        f"fillets.volume_mm3 {fil['volume_mm3']} is not the difference of the two "
        f"published volumes {solid['volume_mm3']} - {solid['volume_nofillet_mm3']}")
    assert fil["volume_mm3"] > 0, (
        f"fillet volume {fil['volume_mm3']} mm^3 is not positive — every junction corner "
        f"is re-entrant, so filleting must ADD material; a negative here means the two "
        f"volumes were measured on different solids")

    # (2) Fillets are a first-order term, not a rounding error.  If this ever drops back
    #     to the 0.01% it was before all forty-eight corners built, the budget below stops
    #     needing the field and this test should be simplified rather than left passing.
    fillet_share = fil["volume_mm3"] / solid["volume_mm3"]
    assert 0.01 < fillet_share < 0.15, f"fillets are {fillet_share:.2%} of the solid"

    # (3) THE BUDGET.  Mesh plus fillets accounts for the solid to within the gusset.
    m = swf.wheel_mass_g(mesh)
    fillet_g = fil["volume_mm3"] * wf.DENSITY_PLA
    gap_g = solid["mass_g_pla"] - (m + fillet_g)
    gap_frac = gap_g / solid["mass_g_pla"]
    assert 0.0 < gap_frac < 0.015, (
        f"mesh {m:.3f} g + fillets {fillet_g:.3f} g leaves {gap_g:+.3f} g "
        f"({gap_frac:+.2%}) against the solid's {solid['mass_g_pla']} g — the gusset is "
        f"the only term left and it is neither positive nor under 1.5%")

    # And it has to be the right SHAPE for a gusset: one per spoke, order 1 mm^2 of
    # section over the full face width.  Measured 1.01 mm^2 at `coarse`, settling to
    # 0.98 at `fine`.
    #
    # ADD BACK THE GUSSET THE MESH NOW MODELS.  Since 2026-08-18 `uncap` continues each
    # junction's far flank to its ring circle, so `wheel_mass_g` already contains 0.234
    # mm² per spoke of what used to be entirely leftover.  The quantity this bound was
    # written about is `_embed`'s WHOLE allowance, so it has to be reassembled from both
    # halves — otherwise the bound silently becomes a different measurement that happens
    # to use the same numbers.  The band is untouched: 0.567 mm²/spoke reassembled here
    # against 0.333 left over, and 0.5 < 0.567 < 2.0 is the same statement as before.
    per_spoke_mm2 = ((gap_g / wf.DENSITY_PLA)
                     / (ww.NUMBER_OF_SPOKES * wf.SPOKE_WIDTH_MM)
                     + ww.area_report(mesh)["gusset_modelled_per_spoke_mm2"])
    assert 0.5 < per_spoke_mm2 < 2.0, (
        f"implied gusset {per_spoke_mm2:.3f} mm^2 per spoke — the leftover is not the "
        f"shape of an embed allowance, so something else is unaccounted for")
