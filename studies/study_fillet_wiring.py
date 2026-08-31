"""
=============================================================================
  WHAT THE REPLACEMENT STRESS TERM CANNOT SEE, AND WHAT IT COSTS —
  PLAN.md §94's ITEMS 3 AND 4, THE LAST TWO BEFORE THE SWITCH
=============================================================================
    .venv-opt/bin/python studies/study_fillet_wiring.py      (make filletwiring)

PLAN.md §95's ranked successor 1.  FILLET_PLAN.md STEP 3.

WHY THIS EXISTS
---------------
§94 proposed `util_j = sigma_fillet_j(mesh) / ALLOWABLE_STRESS_MPA` in place of
`kt * agg`, and put four things in front of it.  §95 measured items 1 and 2 and settled the
quantity: a volume-weighted p-norm with a C2 region weight, `r` = 0.45 mm, `p` = 16.  This
driver is items 3 and 4:

  3.  "`P_c` is excluded by construction and the exclusion must be argued, not assumed.
      Both `P_c` are still re-entrant and divergent, and `rim:P_c` carries the global max
      from `coarse` up.  Excluding them is defensible because they are the END CAP's
      corners and the exported solid has no end cap — §52 ..."
  4.  "The exchange rate does not carry over.  `stress_margin`'s weight was derived at
      `util` = 0.855 under the current construction ...  CHECK: re-derive the weight at
      the new scale before any Stage 3 run, exactly as §15 DEFECT 8 did."

**IT READS SEVEN COMMITTED ARTIFACTS AND SOLVES NOTHING.**  Every number both items need
was measured already, by instruments built for it; a fresh solve here would be an eighth
instrument in a comparison that has to be made on the ones that produced the claims.  The
one thing it computes is a sector's BLOCKING, to count the fillet arcs the mesh actually
makes rather than infer twelve from the spoke count — that number is half of item 3's
hole, and geometry is cheap enough that inferring it would be a choice.

ITEM 3's STATED JUSTIFICATION NAMES A FEATURE THE MESH HAS NOT HAD SINCE 2026-08-18
------------------------------------------------------------------------------------
"They are the END CAP's corners and the exported solid has no end cap" is the pre-uncap
geometry.  `UNCAP_DEFAULT = (True, 1.0)` (PLAN §38, adopted 2026-08-18) **removed the cap
from the MESH too**, and `wheel_wheel.py` says so at the paragraph that explains why it
fillets only `P_t`: *"`P_c` existed because the mesh USED TO close the spoke with a half
END CAP ... SINCE 2026-08-18 `UNCAP_DEFAULT` removes the cap here too, and the second
corner is the far flank's own ring crossing."*  §52 was written five days after the flip
and still called `rim:P_c` "the END CAP's artefact corner"; §94 inherited the phrase and
this driver is what checks it instead of repeating it.

**AT THE HUB THE JUSTIFICATION IS NOT MERELY STALE, IT IS BACKWARDS.**
`study_junction_agreement` reconstructs `wheel_step_export._embed` in numpy and walks the
outline, and its own reconstruction is verified by the crossing count — 24 and 24, the
manifest's `hub_edges` and `rim_edges`.  Against that reconstruction the mesh's `hub:P_c`
is the part's second flank crossing to **0.008 deg of wedge and 1.3e-5 of Williams
exponent**, where the end-cap candidate it replaced is 28.71 deg out.  It is a real
re-entrant corner of the shipped part, at 268.5 deg, and the part FILLETS it: 24 of 24 hub
edges at the full requested radius, `kt_error_pct` 0.0, where the MESH builds one arc per
junction per sector — twelve, at `P_t` only.

At the rim the exclusion holds and for the reason §94 nearly had.  `rim:P_c`'s mesh wedge
is **50.6 deg** from the part's, because `uncap=(True, 1.0)` takes the ring's radial at the
rim rather than `_embed`'s own blend — a deliberate refusal to trade mesh validity for
fidelity, since the faithful blend is accurate to 1.06 deg and collapses
`min_scaled_jacobian` to 0.0072 against a 0.2 barrier.  So the mesh's `rim:P_c` field is
not the part's field, and it is the wheel's global maximum at every rung.

**BOTH EXCLUSIONS SURVIVE.  NEITHER SURVIVES FOR §94's REASON, AND THEY DO NOT SURVIVE FOR
THE SAME REASON AS EACH OTHER.**  The rim's is fidelity: the mesh has a corner the part
does not have there.  The hub's is C1 — the same argument that made `Kt` necessary in the
first place: `hub:P_c` is singular, its peak diverges (9.96 / 15.01 / 18.46 / 22.20 MPa up
the filleted ladder), and a divergent quantity cannot be a constraint whatever it is a
corner of.  Which leaves a hole that has to be named rather than closed here: **a real,
filleted, singular corner of the real part that neither the old term nor the new one
prices.**  It is unpriced today too — `Kt` models `P_t` and `agg` dilutes everything — so
the replacement does not make it worse.  It makes the omission structural instead of
incidental, and that is the honest form of item 3's answer.

ITEM 4: THE WEIGHT DOES NOT MOVE FOR THE REASON §94 GAVE, AND THE KNEE IS THE REAL PROBLEM
--------------------------------------------------------------------------------------------
§15 DEFECT 8 / §23 set `stress_margin`'s weight as an exchange rate: 1% of utilisation
costs 1% of mass at a reference design.  With `stress_margin = w * (util - knee)^2` the
marginal form is

    w  =  mass_term / ( 2 * (util_ref - MARGIN_KNEE_UTIL) * util_ref )

and it reproduces §23's 328.49 at `util_ref` = 0.855 from a mass term of 30.894 — 0.05%
from §18's quoted 30.88, which is what says this is the formula they used and not a
lookalike.  The FINITE 1% step §18 used on the OLD `w * util^2` term gives 304.80 from the
same inputs, 7.2% away, so the two forms are not interchangeable and reproducing 328.49 is
what identifies which was meant.

**NOTHING IN THAT EXPRESSION KNOWS HOW `util` WAS COMPUTED.**  §94's *"a measured nominal
is a different number"* is true of the NUMBER and not of the WEIGHT: the term's shape is
unchanged, `util` is still a utilisation, and at a fixed reference point the only thing
that moves is `mass_term`.  Held at §18's own 0.855 the re-derivation runs 328.49 -> 379.40
— **+15.5%, and none of it from the stress construction**: +9.8% is the filleted mesh
weighing 43.41 g against 39.55, and the rest is the shipped genome having moved since §23
(its mass term is 32.51 today against the 30.894 that 328.49 implies).  A mass measurement
and a promotion, not a consequence of replacing `Kt * agg`.

**WHAT CANNOT BE DONE IS TAKE THE REFERENCE POINT FROM EITHER DESIGN MEASURED HERE, AND
THAT IS ITEM 4's REAL CONTENT.**  §18's 0.855 was "where the design sat" — inside
`[MARGIN_KNEE_UTIL, 1.0]`, above the knee, where the term is live, and at or below the
wall, where `stress` is zero.  Under the replacement both rims are BELOW the knee (0.5067,
0.7845), where the term is inert and no finite weight makes the rate anything, and both
hubs are OVER the wall (1.1415, 1.6760), where a rate derived there is a rate for a place
the optimizer must leave.  **Neither of the two designs this arc has measured can calibrate
the term**, so the reference has to become a POLICY rather than an observation, and the
driver reports the whole curve so the choice is visible rather than buried: at the wall,
`util_ref` = 1.0, the weight is 89.21.

TWO DESIGNS IS NOT THE DESIGN SPACE, and the claim is scoped to them on purpose.  §82's
thirty-two held-out genomes have never been read under the replacement — that is §95's own
successor 2 — so what is established is that the SHIPPED genome and §92's endpoint cannot
supply a reference, not that nothing can.  A genome that lands inside the band would supply
one, and finding whether any does is a measurement nobody has made.

**AND THE KNEE ITSELF, WHICH §94 DID NOT MENTION.**  `MARGIN_KNEE_UTIL = 0.80` is
documented as "a JUDGEMENT about a printed PLA part, not a measurement", and the one piece
of evidence offered for it is *"the shipped genome sits at 0.77952, i.e. essentially AT
this knee ... which is why this restates why that wheel is right rather than condemning
it."*  Under the replacement the shipped genome sits at 1.1415.  **The knee's own stated
justification is gone, and it is a bigger question than the weight**: the weight sets a
rate, the knee decides at what utilisation margin starts costing anything at all, and
0.80 was chosen to land where the design was.

WHAT THIS DRIVER DOES NOT DO
-----------------------------
It does not pick a weight, a knee or a reference point.  Each is a policy this repository
states rather than fits — §18's own comment is that the weight "is an EXCHANGE RATE AND IT
IS A POLICY, SO IT IS STATED" — and the input a policy needs is the curve, which is here.
It does not re-open `ALLOWABLE_STRESS_MPA` or its `FFF_KNOCKDOWN` / `SAFETY_FACTOR`
derivation.  It changes no `src/` module and promotes nothing.

SCOPE.  The corner agreement is GEOMETRY at `coarse` and carries no kinematics.  The
filleted corner ladders are LINEAR at one phase, `study_corner_singularity`'s own.  The
mass terms and today's utilisations are `coarse`, eight phases, both kernels, from
`study_fillet_terms`.  The replacement's utilisations are §95's, LINEAR at one phase.
Those are three different experiments and the tables say which is which.
"""

import argparse
import json
import os

import project_paths as PP  # noqa: F401  (puts src/ on the path)
import _gate_guard

import jax_config  # noqa: F401
import wheel_genome as WG
import wheel_objective as WO
import wheel_wheel as WW

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = PP.ROOT

DEFAULT_INPUTS = {
    "junction": "study_junction_agreement.json",
    "manifest": "../export/wheel_step_manifest.json",
    "corner_shipped": "study_corner_singularity_fillet.json",
    "corner_b029622": "study_corner_singularity_fillet_b029622.json",
    "terms_shipped": "study_fillet_terms.json",
    "terms_b029622": "study_fillet_terms_b029622.json",
    "pnorm": "study_fillet_pnorm.json",
}

GENOMES = (("shipped", "terms_shipped", "corner_shipped"),
           ("b029622", "terms_b029622", "corner_b029622"))
JUNCTIONS = ("hub", "rim")

# The mesh source names `study_junction_agreement` files its rows under.  `SHIPPED DEFAULT`
# is `UNCAP_DEFAULT`; `uncap=False` is the pre-2026-08-18 geometry, kept because it is the
# one §94's justification describes and the comparison is the point.
MESH_AS_BUILT = "mesh (SHIPPED DEFAULT)"
MESH_CAPPED = "mesh (uncap=False)"
PART_SECOND_CROSSING = "bot_flank"
PART_FIRST_CROSSING = "top_flank"

# How close a mesh corner has to sit to the part's before the mesh is reading the part's
# field there.  0.5 deg of wedge, and the number is not a gate anybody tuned: the two
# candidates are 0.008 deg and 50.6 deg apart, so every threshold between 0.02 and 50
# gives the same verdict and the table carries the raw errors anyway.
WEDGE_AGREEMENT_DEG = 0.5

# Reference utilisations for the exchange rate.  Not a sweep of a free parameter — each is
# a POLICY someone could state, and reporting all of them is how the choice stays visible.
#   0.855   §18's own, held fixed so the mass term's move can be seen on its own
#   0.90    the middle of the admissible band [MARGIN_KNEE_UTIL, 1.0]
#   1.00    the wall: "the rate is set where the constraint binds"
# The two designs' own replacement utilisations are added at run time and marked
# inadmissible, because that is item 4's finding rather than a fourth candidate.
POLICY_REFS = (0.855, 0.90, 1.00)

# §23's published weight, reproduced as a self-check on the formula below.
DEFECT8_REF_UTIL = 0.855
DEFECT8_WEIGHT = 328.49
DEFECT8_MASS_TERM_QUOTED = 30.88      # §18's own, quoted to four figures


def _load(path):
    return json.load(open(os.path.join(HERE, path)))


# ---------------------------------------------------------------------------
# ITEM 3 — WHICH CORNERS THE MESH SHARES WITH THE PART
# ---------------------------------------------------------------------------

def corner_agreement(junction):
    """Mesh corner against part corner, per ring, from the committed reconstruction.

    THE RECONSTRUCTION IS VERIFIED BEFORE IT IS BELIEVED.  `study_junction_agreement`
    reimplements `_embed` in numpy rather than calling OCC, so the first thing this reads
    is its crossing count — two per spoke, 24 per wheel per ring — against the shipped
    manifest's `hub_edges` / `rim_edges`.  A reconstruction that found a different number
    of corners than the exporter filleted would be describing a different solid, and
    every wedge angle below would be a number about that solid instead.
    """
    out = {}
    for ring in JUNCTIONS:
        rows = {(c["source"], c["name"]): c for c in junction["rings"][ring]["corners"]}
        part = rows[("part", PART_SECOND_CROSSING)]
        built = rows[(MESH_AS_BUILT, "P_c")]
        capped = rows[(MESH_CAPPED, "P_c")]
        part_t = rows[("part", PART_FIRST_CROSSING)]
        mesh_t = rows[(MESH_CAPPED, "P_t")]
        e = {
            "ring_r_mm": junction["rings"][ring]["ring_r_mm"],
            "part_second_crossing": {k: part[k] for k in
                                     ("theta_deg", "wedge_deg", "lambda_W")},
            "mesh_P_c_as_built": {k: built[k] for k in
                                  ("theta_deg", "wedge_deg", "lambda_W")},
            "mesh_P_c_end_cap": {k: capped[k] for k in
                                 ("theta_deg", "wedge_deg", "lambda_W")},
            "wedge_err_as_built_deg": abs(built["wedge_deg"] - part["wedge_deg"]),
            "wedge_err_end_cap_deg": abs(capped["wedge_deg"] - part["wedge_deg"]),
            "lambda_err_as_built": abs(built["lambda_W"] - part["lambda_W"]),
            "P_t_wedge_err_deg": abs(mesh_t["wedge_deg"] - part_t["wedge_deg"]),
        }
        e["P_c_is_the_parts_corner"] = bool(
            e["wedge_err_as_built_deg"] <= WEDGE_AGREEMENT_DEG)
        out[ring] = e
    return out


def mesh_fillet_arcs_per_wheel(genes, cfg="coarse"):
    """How many fillet arcs `build_wheel(fillet=True)` actually makes, COUNTED.

    One `sector_blocks` call, no solve — the only thing in this driver that is not a read,
    and it is here because the alternative is to infer twelve from `NUMBER_OF_SPOKES` and
    a reading of the blocking.  The number is half of item 3's hole, so it is measured:
    count the `<ring>_fillet_*` block PAIRS in one sector (each pair is one arc, `_a` and
    `_b` sharing the arc between them) and multiply by the sector count.
    """
    b = WW.sector_blocks(genes, WW.get_config(cfg), fillet=True)
    out = {}
    for ring in JUNCTIONS:
        halves = sorted(k for k in b if k.startswith("%s_fillet_" % ring))
        out[ring] = {"blocks": halves, "arcs_per_sector": len(halves) // 2,
                     "arcs_per_wheel": (len(halves) // 2) * WW.NUMBER_OF_SPOKES}
    return out


def what_the_exporter_filleted(manifest, junction, mesh_arcs):
    """The part's filleted edge count against the mesh's fillet-arc count.

    THE ONE FACT ITEM 3's HOLE RESTS ON.  The exporter reports 24 edges found and 24
    filleted per ring at the full requested radius; the reconstruction says 24 is TWO
    crossings per spoke times twelve spokes.  `wheel_wheel` builds one fillet arc per
    junction per sector — `sector_blocks(fillet=True)` yields `<ring>_fillet_a` and
    `_fillet_b`, two blocks forming ONE arc — so the mesh has twelve where the part has
    twenty-four, and the twelve it has are at `P_t`.  That is not an inconsistency to
    fix here; it is the size of what the replacement term cannot see.
    """
    out = {"n_spokes": WW.NUMBER_OF_SPOKES,
           "crossings_per_spoke": junction["crossings_per_spoke"],
           "crossings_per_wheel": junction["crossings_per_wheel"], "rings": {}}
    for row in manifest["fillets"]["detail"]:
        ring = row["junction"]
        out["rings"][ring] = {
            "part_edges_found": row["n_edges_found"],
            "part_edges_filleted": row["n_edges_filleted"],
            "r_requested_mm": row["r_requested_mm"],
            "r_built_mm": row["r_built_mm"],
            "kt_error_pct": row["kt_error_pct"],
            "worst_wedge_deg": row["worst_wedge_deg"],
            "mesh_fillet_arcs": mesh_arcs[ring]["arcs_per_wheel"],
            "mesh_fillet_blocks_per_sector": mesh_arcs[ring]["blocks"],
            "reconstruction_matches_manifest":
                row["n_edges_found"] == junction["crossings_per_wheel"][ring],
        }
    return out


def _diverges(series, tail=3):
    """Increments not shrinking over the finest `tail` rungs.

    `study_corner_singularity`'s own test: a convergent sequence's successive differences
    fall, a divergent one's do not.  Stated as a ratio rather than a threshold so the
    number is in the record — anything at or above 1.0 is a sequence that is not settling.
    """
    d = [series[i + 1] - series[i] for i in range(len(series) - 1)][-tail:]
    ratios = [d[i + 1] / d[i] for i in range(len(d) - 1) if d[i] != 0.0]
    return {"increments": d, "increment_ratios": ratios,
            "diverges": bool(ratios and max(ratios) >= 1.0)}


def excluded_corners(corner, agreement):
    """What the two `P_c` do on the FILLETED mesh, and what they read as utilisations.

    Reported per corner rather than aggregated because the two are excluded for
    DIFFERENT reasons and an aggregate would hide that — the rim's is fidelity, the hub's
    is C1.  The utilisation is `peak / ALLOWABLE_STRESS_MPA` at the finest rung, and it is
    a number to read rather than a constraint to apply: the peak diverges, so the finest
    rung is a lower bound on a quantity that has no limit.
    """
    out = {}
    fine = corner["rungs"][-1]
    for ring in JUNCTIONS:
        name = "%s:P_c" % ring
        series = [r["corners"][name]["peak_vm_mpa"] for r in corner["rungs"]]
        w = corner["williams"].get(name, {})
        out[name] = {
            "peak_mpa": series,
            "wedge_deg": fine["corners"][name]["wedge_deg"],
            "kind": w.get("kind"), "lambda": w.get("lambda"),
            "util_at_fine": series[-1] / WO.ALLOWABLE_STRESS_MPA,
            "is_the_parts_corner": agreement[ring]["P_c_is_the_parts_corner"],
            "wedge_err_vs_part_deg": agreement[ring]["wedge_err_as_built_deg"],
            **_diverges(series),
        }
        out[name]["excluded_because"] = (
            "C1 — the peak diverges, so it cannot be a constraint; and this IS the "
            "part's corner, so the exclusion leaves a real riser unpriced"
            if out[name]["is_the_parts_corner"] else
            "FIDELITY — the mesh's corner is %.1f deg of wedge from the part's, so the "
            "field here is the mesh's and not the wheel's" % out[name][
                "wedge_err_vs_part_deg"])
        # And what the wheel's own global maximum sits on at the finest rung.
        out[name]["carries_global_max"] = bool(
            abs(series[-1] - fine["global_max_vm_mpa"]) < 1e-9)
    return out


# ---------------------------------------------------------------------------
# ITEM 4 — THE EXCHANGE RATE
# ---------------------------------------------------------------------------

def margin_weight(mass_term, util_ref, knee=None):
    """§15 DEFECT 8's exchange rate, as one expression.

        stress_margin = w * (util - knee)^2      so   d/d util = 2 w (util - knee)

    "1% of utilisation costs 1% of mass" is `2 w (u - k) * 0.01 u = 0.01 * mass_term`,
    i.e. `w = mass_term / (2 (u - k) u)`.  The 0.01 cancels, which is why the rate is
    stated as a percentage and derived without one.

    MARGINAL AND NOT A FINITE 1% STEP, and the two are not interchangeable: §18 used the
    finite form on the OLD `w * util^2` term (`w (1.01^2 - 1) util^2`, giving 21.0) and
    §23 used the marginal form on this one.  The finite form here returns 304.9 where §23
    published 328.49, so reproducing that number is what identifies which was meant.

    Returns None at or below the knee, where the derivative is zero and no finite weight
    makes the rate anything — which is not an edge case to guard, it is the statement that
    a design below the knee cannot calibrate a term that is inert there.
    """
    knee = WO.MARGIN_KNEE_UTIL if knee is None else knee
    if util_ref <= knee:
        return None
    return mass_term / (2.0 * (util_ref - knee) * util_ref)


def exchange(terms, pnorm):
    """The rate at every reference point that could be defended, and the two designs'.

    THE ADMISSIBILITY FLAG IS THE FINDING AND NOT A DECORATION.  §18 derived at "where the
    design sits", which was inside `[MARGIN_KNEE_UTIL, 1.0]`.  Under the replacement
    neither design on file is, so the flag is what turns "re-derive at the new scale" from
    an arithmetic task into a question about which policy the reference point encodes.
    """
    rec = pnorm["verdict"]["recommended"]
    rk, pk = "%g" % rec["radius_mm"], "%g" % rec["exponent"]
    out = {"knee": WO.MARGIN_KNEE_UTIL, "wall": 1.0,
           "weight_today": WO.DEFAULT_WEIGHTS["stress_margin"],
           "stress_weight": WO.DEFAULT_WEIGHTS["stress"],
           "allowable_stress_mpa": WO.ALLOWABLE_STRESS_MPA,
           "recommended": {"radius_mm": rec["radius_mm"], "exponent": rec["exponent"]},
           "defect8_reproduction": {}, "genomes": {}}

    # THE SELF-CHECK.  If this does not land on §23's published weight, the formula above
    # is not the one that produced it and nothing below is a re-derivation of anything.
    implied = DEFECT8_WEIGHT * 2.0 * (DEFECT8_REF_UTIL - WO.MARGIN_KNEE_UTIL) \
        * DEFECT8_REF_UTIL
    out["defect8_reproduction"] = {
        "published_weight": DEFECT8_WEIGHT, "ref_util": DEFECT8_REF_UTIL,
        "implied_mass_term": implied,
        "mass_term_quoted_by_18": DEFECT8_MASS_TERM_QUOTED,
        "disagreement_pct": 100.0 * abs(implied / DEFECT8_MASS_TERM_QUOTED - 1.0),
        "weight_from_the_quoted_mass_term":
            margin_weight(DEFECT8_MASS_TERM_QUOTED, DEFECT8_REF_UTIL),
        # THE OTHER FORM, COMPUTED RATHER THAN ASSERTED.  §18 set the OLD `w * util^2`
        # term with a FINITE 1% step; §23 set this one with the marginal rate.  Landing
        # 7% away from the published weight is what says the two are not interchangeable,
        # and it is the kind of number that has no business being arithmetic in prose.
        "weight_from_the_finite_1pct_step": 0.01 * implied / (
            (1.01 * DEFECT8_REF_UTIL - WO.MARGIN_KNEE_UTIL) ** 2
            - (DEFECT8_REF_UTIL - WO.MARGIN_KNEE_UTIL) ** 2),
    }

    for gname, tkey, _ in GENOMES:
        t = terms[tkey]
        row = {"mass_term": {}, "mass_g": {}, "util_today": {}, "util_replacement": {},
               "weight_at_ref": {}, "barrier_at_replacement_util": {}}
        for mesh in ("unfilleted", "filleted"):
            e = t["evaluation"]["svk"][mesh]
            row["mass_term"][mesh] = e["terms"]["mass"]["value"]
            row["mass_g"][mesh] = e["report"]["mesh_mass_g"]
            row["util_today"][mesh] = {
                j: e["report"]["stress_utilisation_%s" % j] for j in JUNCTIONS}
        mass = row["mass_term"]["filleted"]        # the replacement only exists there
        for j in JUNCTIONS:
            v = pnorm["convergence"][gname][j]["radii"][rk]["bump3"][
                "exponents"][pk]["values_mpa"][-1]
            row["util_replacement"][j] = v / WO.ALLOWABLE_STRESS_MPA

        refs = list(POLICY_REFS) + [row["util_replacement"][j] for j in JUNCTIONS]
        for u in sorted(set(round(x, 6) for x in refs)):
            row["weight_at_ref"]["%g" % u] = {
                "util_ref": u,
                "weight_filleted_mass": margin_weight(mass, u),
                "weight_unfilleted_mass": margin_weight(
                    row["mass_term"]["unfilleted"], u),
                # ADMISSIBLE means "a design could sit here and ship": above the knee, so
                # the term is live, and at or below the wall, so `stress` is zero.
                "admissible_reference": bool(WO.MARGIN_KNEE_UTIL < u <= 1.0),
                "is_a_design": any(abs(u - row["util_replacement"][j]) < 1e-6
                                   for j in JUNCTIONS),
            }
        for j in JUNCTIONS:
            u = row["util_replacement"][j]
            row["barrier_at_replacement_util"][j] = {
                "util": u,
                "stress": float(WO.soft_barrier(
                    u - 1.0, WO.DEFAULT_WEIGHTS["stress"])),
                "stress_margin_today": float(WO.soft_barrier(
                    u - WO.MARGIN_KNEE_UTIL, WO.DEFAULT_WEIGHTS["stress_margin"])),
            }
        row["mass_term_change_pct"] = 100.0 * (
            mass / row["mass_term"]["unfilleted"] - 1.0)
        row["any_design_admissible"] = any(
            WO.MARGIN_KNEE_UTIL < row["util_replacement"][j] <= 1.0 for j in JUNCTIONS)
        out["genomes"][gname] = row
    return out


def knee_justification(terms, exch):
    """`MARGIN_KNEE_UTIL`'s own evidence, re-read at the new construction.

    §94's item 4 is about the WEIGHT and does not mention the knee.  The knee is the
    larger of the two: the weight sets a rate, the knee decides at what utilisation
    margin starts costing anything at all — and the only evidence `wheel_objective` offers
    for 0.80 is that the shipped genome sits essentially on it.  Under the replacement it
    does not, so the sentence that justifies the constant stops being true about the
    design it was justified on.
    """
    ship = exch["genomes"]["shipped"]
    today = ship["util_today"]["unfilleted"]["hub"]
    repl = ship["util_replacement"]["hub"]
    return {
        "knee": WO.MARGIN_KNEE_UTIL,
        "documented_evidence": "the shipped genome sits at 0.77952, essentially AT this "
                               "knee (wheel_objective.MARGIN_KNEE_UTIL's comment)",
        "shipped_hub_util_today_unfilleted": today,
        "shipped_hub_util_today_filleted":
            ship["util_today"]["filleted"]["hub"],
        "shipped_hub_util_replacement": repl,
        "distance_from_knee_today": today - WO.MARGIN_KNEE_UTIL,
        "distance_from_knee_replacement": repl - WO.MARGIN_KNEE_UTIL,
        "evidence_survives": bool(abs(repl - WO.MARGIN_KNEE_UTIL) < 0.05),
    }


# ---------------------------------------------------------------------------
# BUILD AND REPORT
# ---------------------------------------------------------------------------

def build(inputs=None):
    inputs = dict(DEFAULT_INPUTS if inputs is None else inputs)
    art = {k: _load(v) for k, v in inputs.items()}
    rep = {"inputs": inputs, "solves": 0,
           "wedge_agreement_gate_deg": WEDGE_AGREEMENT_DEG}

    rep["agreement"] = corner_agreement(art["junction"])
    genes = json.load(open(os.path.join(ROOT, "best_solution.json")))["genes"]
    rep["mesh_fillet_arcs"] = mesh_fillet_arcs_per_wheel(
        [genes[n] for n in WG.GENE_NAMES])
    rep["what_the_exporter_filleted"] = what_the_exporter_filleted(
        art["manifest"], art["junction"], rep["mesh_fillet_arcs"])
    rep["excluded"] = {
        g: excluded_corners(art[ck], rep["agreement"]) for g, _, ck in GENOMES}
    rep["exchange"] = exchange({k: art[k] for _, k, _ in GENOMES}, art["pnorm"])
    rep["knee"] = knee_justification({k: art[k] for _, k, _ in GENOMES}, rep["exchange"])

    a = rep["agreement"]
    rep["verdict"] = {
        # ITEM 3
        "end_cap_justification_is_stale": True,
        "hub_P_c_is_the_parts_corner": a["hub"]["P_c_is_the_parts_corner"],
        "rim_P_c_is_the_parts_corner": a["rim"]["P_c_is_the_parts_corner"],
        "hub_wedge_err_as_built_deg": a["hub"]["wedge_err_as_built_deg"],
        "hub_wedge_err_end_cap_deg": a["hub"]["wedge_err_end_cap_deg"],
        "rim_wedge_err_as_built_deg": a["rim"]["wedge_err_as_built_deg"],
        "both_P_c_diverge_on_the_filleted_mesh": all(
            rep["excluded"][g]["%s:P_c" % j]["diverges"]
            for g, _, _ in GENOMES for j in JUNCTIONS),
        "the_part_fillets_what_the_mesh_does_not": all(
            r["part_edges_filleted"] > r["mesh_fillet_arcs"]
            for r in rep["what_the_exporter_filleted"]["rings"].values()),
        # ITEM 4
        "weight_is_invariant_to_how_util_is_computed": True,
        "mass_term_change_pct": rep["exchange"]["genomes"]["shipped"][
            "mass_term_change_pct"],
        "neither_measured_design_is_an_admissible_reference": not any(
            rep["exchange"]["genomes"][g]["any_design_admissible"]
            for g, _, _ in GENOMES),
        "knee_evidence_survives": rep["knee"]["evidence_survives"],
    }
    return rep


def _bar(t):
    print("\n" + "=" * 78 + "\n  " + t + "\n" + "=" * 78)


def _print(rep):
    _bar("A  ITEM 3 — WHICH JUNCTION CORNERS THE MESH SHARES WITH THE PART")
    f = rep["what_the_exporter_filleted"]
    print("    The reconstruction is verified before it is believed: %d crossings per"
          % f["crossings_per_spoke"]["hub"])
    print("    spoke, %d per wheel per ring, against the manifest's own edge count.\n"
          % f["crossings_per_wheel"]["hub"])
    print("    ring   part edges  filleted   R_req    R_built   Kt err   mesh arcs   "
          "recon == manifest")
    for ring, r in sorted(f["rings"].items()):
        print("    %-6s %8d %10d  %7.4f  %7.4f  %6.1f%%   %8d   %s"
              % (ring, r["part_edges_found"], r["part_edges_filleted"],
                 r["r_requested_mm"], r["r_built_mm"], r["kt_error_pct"],
                 r["mesh_fillet_arcs"], r["reconstruction_matches_manifest"]))
    print("\n    THE PART FILLETS BOTH CROSSINGS AT EACH JUNCTION.  The mesh builds one")
    print("    arc per junction per sector, at `P_t`.\n")

    print("    corner                       theta      wedge     lambda    vs part")
    for ring in JUNCTIONS:
        e = rep["agreement"][ring]
        print("    %s:  ring r %.2f mm" % (ring, e["ring_r_mm"]))
        for label, key, err in (
                ("part 2nd flank crossing", "part_second_crossing", None),
                ("mesh P_c AS BUILT", "mesh_P_c_as_built", "wedge_err_as_built_deg"),
                ("mesh P_c end cap (pre-38)", "mesh_P_c_end_cap",
                 "wedge_err_end_cap_deg")):
            c = e[key]
            print("      %-27s %9.5f %9.4f  %9.6f  %s"
                  % (label, c["theta_deg"], c["wedge_deg"], c["lambda_W"],
                     "—" if err is None else "%9.4f deg" % e[err]))
        print("      => `%s:P_c` IS the part's corner: %s   (P_t agrees to %.4f deg)"
              % (ring, e["P_c_is_the_parts_corner"], e["P_t_wedge_err_deg"]))

    _bar("B  ITEM 3 — WHAT THE EXCLUDED CORNERS DO ON THE FILLETED MESH")
    print("    linear, one phase, smoke..fine.  `util` is peak / %.1f MPa at the finest"
          % WO.ALLOWABLE_STRESS_MPA)
    print("    rung, and it is a LOWER BOUND on a quantity that has no limit.\n")
    print("    genome   corner      wedge    lambda    peak MPa up the ladder"
          "               ratios    util@fine  global max")
    for g, _, _ in GENOMES:
        for name, e in sorted(rep["excluded"][g].items()):
            print("    %-8s %-9s %8.3f  %.4f  %s   %s   %8.4f   %s"
                  % (g, name, e["wedge_deg"], e["lambda"] or float("nan"),
                     " ".join("%8.3f" % v for v in e["peak_mpa"]),
                     " ".join("%5.3f" % r for r in e["increment_ratios"]),
                     e["util_at_fine"], e["carries_global_max"]))
    print()
    for g, _, _ in GENOMES:
        for name, e in sorted(rep["excluded"][g].items()):
            if g != "shipped":
                continue
            print("    %-9s excluded because %s" % (name, e["excluded_because"]))

    _bar("C  ITEM 4 — THE EXCHANGE RATE, RE-DERIVED")
    d = rep["exchange"]["defect8_reproduction"]
    print("    w = mass_term / (2 (u - knee) u),  knee = %.2f\n"
          % rep["exchange"]["knee"])
    print("    SELF-CHECK against §23's published %.2f at u = %.3f: it implies a mass"
          % (d["published_weight"], d["ref_util"]))
    print("    term of %.4f against §18's quoted %.2f — %.3f%% apart, so this is the"
          % (d["implied_mass_term"], d["mass_term_quoted_by_18"], d["disagreement_pct"]))
    print("    formula that produced it — the FINITE 1% step §18 used on the old term")
    print("    gives %.2f from the same inputs, %.1f%% away.  (The quoted mass term alone"
          % (d["weight_from_the_finite_1pct_step"],
             100.0 * abs(d["weight_from_the_finite_1pct_step"]
                         / d["published_weight"] - 1.0)))
    print("    gives %.2f.)\n" % d["weight_from_the_quoted_mass_term"])
    for g, _, _ in GENOMES:
        r = rep["exchange"]["genomes"][g]
        print("    %s:  mass term %.4f unfilleted (%.4f g) -> %.4f filleted (%.4f g), "
              "%+.1f%%"
              % (g, r["mass_term"]["unfilleted"], r["mass_g"]["unfilleted"],
                 r["mass_term"]["filleted"], r["mass_g"]["filleted"],
                 r["mass_term_change_pct"]))
        print("      util today (svk, filleted mesh): hub %.4f  rim %.4f"
              % (r["util_today"]["filleted"]["hub"], r["util_today"]["filleted"]["rim"]))
        print("      util under the replacement:      hub %.4f  rim %.4f"
              % (r["util_replacement"]["hub"], r["util_replacement"]["rim"]))
        print("      u_ref   w (filleted mass)  w (unfilleted)  admissible ref  "
              "is a design")
        for k in sorted(r["weight_at_ref"], key=float):
            w = r["weight_at_ref"][k]
            print("      %6.4f  %17s  %14s  %-14s  %s"
                  % (w["util_ref"],
                     "inert" if w["weight_filleted_mass"] is None
                     else "%.2f" % w["weight_filleted_mass"],
                     "inert" if w["weight_unfilleted_mass"] is None
                     else "%.2f" % w["weight_unfilleted_mass"],
                     "yes" if w["admissible_reference"] else "NO",
                     "yes" if w["is_a_design"] else ""))
        print("      NEITHER JUNCTION OF THIS DESIGN IS AN ADMISSIBLE REFERENCE: %s"
              % (not r["any_design_admissible"]))
        print("      what the raw substitution fires at today's weights:")
        for j in JUNCTIONS:
            b = r["barrier_at_replacement_util"][j]
            print("        %-4s util %.4f   stress %10.4f   stress_margin %10.4f"
                  % (j, b["util"], b["stress"], b["stress_margin_today"]))
        print()

    _bar("D  ITEM 4's LARGER HALF — THE KNEE'S OWN EVIDENCE")
    k = rep["knee"]
    print("    MARGIN_KNEE_UTIL = %.2f, and the evidence `wheel_objective` offers is:"
          % k["knee"])
    print("      \"%s\"\n" % k["documented_evidence"])
    print("      shipped hub util today, unfilleted mesh   %.4f   (%+.4f from the knee)"
          % (k["shipped_hub_util_today_unfilleted"], k["distance_from_knee_today"]))
    print("      shipped hub util today, filleted mesh     %.4f"
          % k["shipped_hub_util_today_filleted"])
    print("      shipped hub util under the replacement    %.4f   (%+.4f from the knee)"
          % (k["shipped_hub_util_replacement"], k["distance_from_knee_replacement"]))
    print("\n      THE EVIDENCE SURVIVES: %s" % k["evidence_survives"])
    print("      The weight sets a RATE; the knee decides at what utilisation margin")
    print("      starts costing anything at all.  0.80 was chosen to land where the")
    print("      design was, and under the replacement the design is not there.")

    _bar("THE VERDICT")
    for k_, v in rep["verdict"].items():
        print("    %-45s %s" % (k_, v if not isinstance(v, float) else "%.4f" % v))
    print("\n    ITEM 3: both exclusions survive, NEITHER for §94's reason, and not for")
    print("    the same reason as each other.  The rim's is fidelity; the hub's is C1,")
    print("    and it leaves a real filleted singular corner of the real part unpriced")
    print("    — unpriced today too, but structurally rather than incidentally.")
    print("    ITEM 4: the weight is invariant to how `util` is computed; what moves is")
    print("    the mass term (+%.1f%%) and, decisively, the REFERENCE POINT, which can no"
          % rep["verdict"]["mass_term_change_pct"])
    print("    longer be taken from a design.  And the knee's own evidence is gone.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for k, v in DEFAULT_INPUTS.items():
        ap.add_argument("--%s" % k.replace("_", "-"), default=v)
    ap.add_argument("--out", default="study_fillet_wiring.json")
    args = ap.parse_args()

    inputs = {k: getattr(args, k) for k in DEFAULT_INPUTS}
    # This driver solves nothing, so there is no fidelity flag to lower — the only way to
    # weaken it is to point it at a weaker measurement, and then the verdict is about that
    # measurement instead.  Same shape as `study_fillet_kt`'s guard, same reason.
    _gate_guard.refuse_degraded_out(ap, args, "study_fillet_wiring.json", [
        (inputs[k] != v, "--%s %s, not the committed %s"
                         % (k.replace("_", "-"), inputs[k], v))
        for k, v in DEFAULT_INPUTS.items()])

    rep = build(inputs)
    _print(rep)
    with open(os.path.join(HERE, args.out), "w") as fh:
        json.dump(rep, fh, indent=2)
    print("\n    wrote %s" % args.out)

    # NO PASS/FAIL AND EXIT 0, for `study_fillet_kt`'s reason.  §94 asked for an ARGUMENT
    # and a re-derivation; neither has a threshold to meet, and the verdict's booleans are
    # the findings — including the two that overturn the section that asked.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
