"""
Pins for `studies/study_junction_agreement.py`'s fillet-fit pricing — FILLET_PLAN.md
STEP 1 RECORD PART 8, PLAN.md §46.

WHY THIS FILE EXISTS. That driver had no test, and it now carries the number the fillet
arc's go/no-go rests on: whether a fillet of the shipped radius can sit in each junction
corner at all. FILLET_PLAN PART 2 answered that once, on 2026-08-17, and ruled the second
corner `P_c` out permanently on a spoke-side leg of `t/2`. **Every term in that judgement
has since moved** — the void angle, the leg, and which geometry is the default — because
§38 replaced the end cap. The verdict is now recomputed by the driver rather than quoted,
and pinned here.

WHAT IS PINNED. The ARITHMETIC against closed-form cases, the DIRECTION of each verdict
with the mechanism that sets it, and — the §45 pattern — that the committed artifact still
describes the mesh the tree builds today. Not the millimetres: `r_max_on_this_leg_mm` is a
function of a mesh option and a gene, and pinning it tightly would assert as a property of
the wheel something that is a property of a setting.

GEOMETRIC ADMISSIBILITY IS NOT MESHABILITY. A corner that accepts a fillet on its legs may
still be unbuildable by the current construction — `make fillet` and
`tests/test_fillet_fold.py` are that gate, and it is far tighter (0.12-0.24 mm at
`coarse`). Nothing here says a fillet can be MESHED.
"""

import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import wheel_genome as wg                  # noqa: E402
import wheel_wheel as ww                   # noqa: E402
import study_junction_agreement as ja      # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def genes():
    with open(os.path.join(REPO, "best_solution.json")) as fh:
        return json.load(fh)["genes"]


@pytest.fixture(scope="module")
def report():
    with open(os.path.join(REPO, "studies", "study_junction_agreement.json")) as fh:
        return json.load(fh)


def _corner(report, ring, source, name):
    for row in report["rings"][ring]["corners"]:
        if row["source"] == source and row["name"] == name:
            return row
    raise KeyError(f"{ring} {source} {name}")


# ---------------------------------------------------------------------------
# THE ARITHMETIC, AGAINST CASES WITH KNOWN ANSWERS
# ---------------------------------------------------------------------------

def test_the_tangent_length_is_right_where_the_answer_is_known():
    """`T = R / tan(void/2)`, checked at the two voids where it is exact by inspection.

    A 90 deg void puts the tangent length equal to the radius; a 60 deg void makes it
    `R * sqrt(3)`. Without these the pricing is an unchecked formula, and it is the
    formula the whole `P_c` go/no-go turns on.
    """
    assert ja.fillet_fit(90.0, 1.0, 1.0)["tangent_length_mm"] == pytest.approx(1.0)
    assert ja.fillet_fit(60.0, 1.0, 1.0)["tangent_length_mm"] == pytest.approx(math.sqrt(3.0))
    assert ja.fillet_fit(120.0, 1.0, 1.0)["tangent_length_mm"] == pytest.approx(
        1.0 / math.sqrt(3.0))


def test_r_max_inverts_the_tangent_length_exactly():
    """`r_max_on_this_leg_mm` must be the radius whose tangent length IS the leg.

    Checked as a round trip rather than as a second formula, because a sign or a factor of
    two in the half-angle would pass a spot check and fail here.
    """
    for void in (30.0, 62.8, 89.5, 91.5, 141.2):
        leg = 0.75
        r_max = ja.fillet_fit(void, leg, 1.0)["r_max_on_this_leg_mm"]
        assert ja.fillet_fit(void, leg, r_max)["tangent_length_mm"] == pytest.approx(leg)
        assert ja.fillet_fit(void, leg, r_max * 1.001)["fits"] is False
        assert ja.fillet_fit(void, leg, r_max * 0.999)["fits"] is True


def test_no_leg_means_no_verdict_rather_than_a_default_one():
    """The PART's corners are read off an outline and have no measured leg.

    Reporting `fits` for them would be inventing a judgement from a missing input — the
    failure mode this file exists to prevent one level up.
    """
    priced = ja.fillet_fit(90.0, None, 1.0)
    assert "tangent_length_mm" in priced
    assert "fits" not in priced and "t_over_leg" not in priced


# ---------------------------------------------------------------------------
# THE VERDICTS, AND THE MECHANISM BEHIND EACH
# ---------------------------------------------------------------------------

def test_P_t_accepts_a_fillet_with_room_to_spare_at_both_rings(report):
    """`P_t` lies on the spoke's own flank, so its leg is the whole 41 mm span.

    This is why the arc's construction fillets `P_t` and nothing else, and it does not
    move with `uncap` — `P_t` is the spoke block's end row.
    """
    for ring in ("hub", "rim"):
        fit = _corner(report, ring, "mesh (uncap=False)", "P_t")["fillet_fit"]
        assert fit["fits"] is True
        assert fit["t_over_leg"] < 0.25
        assert fit["spoke_side_leg_mm"] > 40.0


def test_the_end_cap_refused_the_fillet_at_both_rings(report):
    """PART 2's original NO-GO, kept as the control: on the capped mesh the leg is `t/2`.

    0.737 mm at the hub and 0.716 at the rim, against tangent lengths of 1.09 and 6.07.
    That geometry is no longer the default and this row is the `uncap=False` control.
    """
    for ring, expect in (("hub", 1.4), ("rim", 8.0)):
        fit = _corner(report, ring, "mesh (uncap=False)", "P_c")["fillet_fit"]
        assert fit["fits"] is False
        assert fit["t_over_leg"] > expect
        assert fit["spoke_side_leg_mm"] < 0.8


def test_uncapping_FLIPPED_the_hub_verdict_and_did_not_flip_the_rim(report):
    """THE FINDING. The same change moves the two rings in the same direction by
    different amounts, and only one of them crosses.

    Uncapping opens `P_c` from a 63/53 deg void to 92/89, and a wider void needs a
    SHORTER tangent — so the fillet gets easier even though the leg itself got shorter
    (0.737 -> 0.660 at the hub, 0.716 -> 0.566 at the rim). At the hub that is enough:
    `T/leg` falls 1.47 -> 0.98 and the shipped radius fits. At the rim it is not: 8.49 ->
    5.34, and the shipped `R_rim` of 3.0 mm needs a leg five times what is there.

    **The rim is the one that matters**, because the wheel's global peak sits on
    `rim:P_c` (FILLET_PLAN PART 4, re-measured in PART 7). Pinned as the pair, since the
    interesting claim is the contrast rather than either number.
    """
    hub = _corner(report, "hub", "mesh (SHIPPED DEFAULT)", "P_c")["fillet_fit"]
    rim = _corner(report, "rim", "mesh (SHIPPED DEFAULT)", "P_c")["fillet_fit"]
    assert hub["fits"] is True and hub["t_over_leg"] < 1.0
    assert rim["fits"] is False and rim["t_over_leg"] > 4.0

    capped_hub = _corner(report, "hub", "mesh (uncap=False)", "P_c")["fillet_fit"]
    capped_rim = _corner(report, "rim", "mesh (uncap=False)", "P_c")["fillet_fit"]
    assert hub["t_over_leg"] < capped_hub["t_over_leg"]
    assert rim["t_over_leg"] < capped_rim["t_over_leg"]
    # both legs got SHORTER; the verdicts improved anyway, which is what says the void
    # angle is what moved the answer rather than the leg
    assert hub["spoke_side_leg_mm"] < capped_hub["spoke_side_leg_mm"]
    assert rim["spoke_side_leg_mm"] < capped_rim["spoke_side_leg_mm"]


def test_the_faithful_rim_would_buy_a_factor_of_FOUR_on_the_admissible_radius(report):
    """§37 filed the rim tri-block as buying "only rim corner fidelity — not convergence,
    not the fillet, not a quotable peak". The middle clause is what this pins against.

    `uncap=True` at the rim is the faithful geometry (1.06 deg of wedge against the part,
    versus 50.61 as built) and it is the one §36 measured as unbuildable without the
    tri-block, at `min_sj` 0.0072. On that geometry `r_max_on_this_leg_mm` goes from 0.56
    mm to 2.59 — **a factor of 4.6, and within 14% of the shipped `R_rim` of 3.0.**

    So the tri-block does buy the fillet at the corner that carries the peak. It does not
    buy it AT THE SHIPPED RADIUS, and that distinction is the whole of the ranking: see
    PLAN §46. Pinned as a ratio and a direction, not as millimetres.
    """
    built = _corner(report, "rim", "mesh (SHIPPED DEFAULT)", "P_c")["fillet_fit"]
    faithful = _corner(report, "rim", "mesh (uncap=True)", "P_c")["fillet_fit"]
    assert faithful["r_max_on_this_leg_mm"] > 4.0 * built["r_max_on_this_leg_mm"]
    assert faithful["fits"] is False, (
        "the faithful rim now admits the SHIPPED R_rim — that would retire PLAN §46's "
        "central qualifier and the ranking that rests on it")


# ---------------------------------------------------------------------------
# THE §45 CHECK: DOES THE COMMITTED ARTIFACT STILL DESCRIBE THE TREE?
# ---------------------------------------------------------------------------

def test_the_committed_report_describes_the_mesh_the_tree_BUILDS_TODAY(genes, report):
    """This driver reads `WW.UNCAP_DEFAULT` at run time, so its "SHIPPED DEFAULT" rows
    mean whatever the default was when it ran.

    §45 found the sibling artifact (`study_corner_singularity.json`) four days stale
    against exactly that mechanism, with nothing red because every test read the same
    stale file. Same guard here, and it is a re-measurement rather than a stored
    comparison: rebuild sector 0's corners under today's default and check the report
    still describes them.
    """
    gvec = wg.genes_to_vector(genes)
    fresh = ja.mesh_corners(gvec, report["config"], uncap=ww.UNCAP_DEFAULT)
    for ring in ("hub", "rim"):
        row = _corner(report, ring, "mesh (SHIPPED DEFAULT)", "P_c")
        e = fresh[ring]["P_c"]
        void, wedge = ja.void_and_wedge(e["point"], e["f_hat"], e["away"])
        assert wedge == pytest.approx(row["wedge_deg"], abs=1e-6), (
            f"{ring}: today's default builds a {wedge:.3f} deg wedge and the committed "
            f"report says {row['wedge_deg']:.3f} — re-run `make junction`")
        assert e["leg_mm"] == pytest.approx(row["spoke_side_leg_mm"], rel=1e-9)
