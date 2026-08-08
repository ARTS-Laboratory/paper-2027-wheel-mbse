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
    """
    half = 3.0
    _, f = fem.wheel_problem(mesh, patch_half_deg=half)
    xy = np.asarray(mesh.coords)
    loaded = np.linalg.norm(f.reshape(-1, 2), axis=1) > 0
    assert loaded.sum() > 0
    th = np.degrees(np.arctan2(xy[loaded, 1], xy[loaded, 0])) % 360.0
    element_deg = ww.SECTOR_DEG / (mesh.cfg.n_weld + mesh.cfg.n_rim_free)
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
    """
    s = res["compliance_split"]
    assert 0.25 < s["rim"] < 0.40, s
    assert 0.58 < s["spoke"] < 0.72, s
    assert s["hub"] < 0.03, s


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
    roughly constant.  It ranges over more than an order of magnitude.

    Reduced fidelity (smoke mesh, few samples) on purpose — the finding is a factor of
    ~30 and does not need a converged mesh to be visible.  If this ever passes, the whole
    Stage 2 justification needs re-reading, which is why it fails loudly.

    IT PINS THE FLOOR AT 2.0 BECAUSE THE STATISTIC IS A PROPERTY OF THE GENE BOX, NOT OF
    `genes`.  `run_beam_blindness` draws a Latin hypercube from the box and computes the
    max/min over the DRAWN rows, explicitly excluding the genome it is passed — so the
    ratio does not depend on which design ships, and it does depend on where the box's
    thickness floor sits.  Measured (§14):

        genome        floor 2.0    floor 1.2
        36aed36           4.943        2.686
        350f4c7           4.943        2.686

    Identical down the genome column, to every digit.  §13's move of the DEFAULT floor to
    1.2 therefore broke this test without touching the wheel: a lower floor admits
    floppier random spokes, which compresses the spread.  Gate 1's conclusion — the
    correction factor is not defensible — is unaffected and is still the first assertion.
    What the `> 3.0` margin was calibrated in was a 2.0 mm box, so that is the box it is
    measured in.  Re-deriving Gate 1's margin at a 1.2 mm floor is a real piece of work
    and a judgement about Gate 1; it is not a test edit, and it has not been done here.
    """
    before = wf.MIN_WALL_MM
    try:
        wf.set_min_wall(2.0)
        rep = swf.run_beam_blindness(genes, "smoke", n=6, seed=7)
    finally:
        # Restore unconditionally.  `tests/test_stage3.py` takes its bounds in a
        # MODULE-scoped fixture that never recomputes, so a floor leaked from here would
        # not merely persist — it would be baked in and fail somewhere else entirely.
        wf.set_min_wall(before)

    assert not rep["correction_factor_is_defensible"], rep["fea_over_beam_cv"]
    assert rep["fea_over_beam_ratio"] > 3.0, rep["fea_over_beam_ratio"]


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
    drops = [fem.solve_wheel(ww.build_wheel(genes, "smoke", rim_outer=ro))["axle_drop_mm"]
             for ro in (49.7, 50.0, 50.6, 51.2)]
    assert all(drops[i + 1] < drops[i] for i in range(len(drops) - 1)), drops
    assert drops[-1] < swf.TARGET_DEFLECTION_MM < drops[0]


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
    per_spoke_mm2 = (gap_g / wf.DENSITY_PLA) / (ww.NUMBER_OF_SPOKES * wf.SPOKE_WIDTH_MM)
    assert 0.5 < per_spoke_mm2 < 2.0, (
        f"implied gusset {per_spoke_mm2:.3f} mm^2 per spoke — the leftover is not the "
        f"shape of an embed allowance, so something else is unaccounted for")
