"""
=============================================================================
  COMPLIANT WHEEL — REQUIREMENTS (mission -> constants, points -> weights)
=============================================================================
The layer this tree spent ninety-six numbered sections without.  MBSE_PLAN.md.

WHAT THIS FILE IS FOR
---------------------
The whole mission — the vehicle, the arrival, the environment, the service life — used
to be two bare literals in `wheel_fea.py`:

    FORCE_LBS            = 15.0   (:151)   -> 66.7233 N, and that is the WHOLE vehicle
    TARGET_DEFLECTION_MM = 2.0    (:156)   -> the WHOLE stroke requirement

Neither carried a derivation, and their neighbours all do: `MIN_WALL_MM` carries eighteen
lines on why 1.2, `CY_BOUND_MM` carries a three-row sweep proving the bound is not
binding, `R_hub`'s floor carries twenty-five lines on one extrusion width.  This module
is the derivation those two never had, plus the two axes nothing in the tree could reach
at all — ambient temperature and service life.

THE SPLIT IS THE DESIGN, AND IT IS NOT A UI CONVENTION
------------------------------------------------------
A requirement you cannot choose is not a preference, and a preference is not a
requirement.  There are two input surfaces and they never mix:

    MISSION — absolute facts, entered as numbers    PRIORITIES — 100 points, zero-sum
    -----------------------------------------      ---------------------------------
      auw_kg, n_wheels, k_asym, sink_rate  -> force   light        -> mass
      field_class                          -> stroke  soft landing -> deflection
      ambient_c        -> e_mpa, allowable_stress     durability   -> stress_margin
      landings         -> safety_factor -> allowable  rolling      -> phase_ripple
      nozzle_mm, perimeters                -> min_wall print finish -> smoothness

`PRIORITY_AXES` is `wheel_objective.OBJECTIVE_TERMS`, READ rather than retyped: the same
five names in the same order.  Points move `should`s.  **Points never reach
`BARRIER_TERMS`** — you cannot buy your way out of a mesh that does not integrate, a
spoke that folds through itself, or a fillet that does not fit in its sector.  Those are
`shall`s and their only admissible value is zero, which `wheel_objective.py:394-401`
already asserts and this module reuses rather than re-inventing.

WHAT THIS FILE DELIBERATELY CANNOT REACH
-----------------------------------------
**Ø100 is frozen.**  Ground clearance and prop clearance are real requirements and they
want `RIM_RADIUS_MM`; `wheel_fea.py:113-137` states the price — changing it
*"REINTERPRETS every gene on disk"* — so every axis here leaves the genome frame intact.
That is what lets `best_solution.json` and `tests/test_golden.py` stay meaningful while
requirements move.  `NUMBER_OF_SPOKES = 12` is likewise not a parameter: it is baked into
`SECTOR_DEG`, the mesh's twelve-fold periodicity and the `/3` in
`FORCE_PER_SPOKE_NEWTONS`.

**There is no `runway_m`, and its absence is deliberate.**  MBSE_PLAN.md names runway
length as a mission axis reaching `sink_rate` through approach angle.  Nothing in this
tree measures that correlation, and the FEA applies a **purely radial** load on **flat,
rigid, frictionless** ground (`wheel_fem.wheel_contact_problem:1717`,
`RigidGroundContact:652`) — no braking load, no side load, no obstacle bump, no rolling
friction, no tyre.  A field-length axis whose only route into the physics is an
unmeasured correlation into a number the caller can already type is a wish with a
units label, so `sink_rate_ms` is entered directly and `field_class` reaches the stroke
and nothing else.

IMPORT HYGIENE — numpy + stdlib, and `wheel_objective` ONLY LAZILY
------------------------------------------------------------------
Mirrors `wheel_genome.py`'s contract: the CadQuery interpreter must be able to import
this, and that env has numpy but not jax.  `DEFAULT_WEIGHTS`, `OBJECTIVE_TERMS` and
`MARGIN_KNEE_UTIL` live in `wheel_objective`, which imports jax at module scope, so every
read of them goes through `_objective_module()` inside a function body.  A bare
`import wheel_requirements` stays jax-free and `tests/test_import_hygiene.py` is the
tripwire.  The mission derivations — force, stroke, allowable, min wall — need none of
it and work in the CAD env.
=============================================================================
"""

import hashlib
import json
from dataclasses import dataclass, field, replace

import numpy as np

import wheel_fea as W

# ---------------------------------------------------------------------------
# THE LAZY DOOR TO `wheel_objective`
# ---------------------------------------------------------------------------


def _objective_module():
    """`wheel_objective`, imported at CALL time and never at module scope.

    See the module docstring's hygiene note.  Everything this module needs from there —
    `DEFAULT_WEIGHTS`, `OBJECTIVE_TERMS`, `BARRIER_TERMS`, `MARGIN_KNEE_UTIL` — is read
    through here, so the set of things that would break a CAD-env import is exactly the
    set of functions that call this, and `priority_axes()` says which those are.
    """
    import wheel_objective
    return wheel_objective


def _default_weights():
    """`wheel_objective.DEFAULT_WEIGHTS`, through the lazy door.  A fresh dict each call:
    the caller must not be able to mutate the table every weight in the tree reads."""
    return dict(_objective_module().DEFAULT_WEIGHTS)


def priority_axes():
    """The five point axes, WHICH ARE `wheel_objective.OBJECTIVE_TERMS`, in its order.

    Read rather than retyped so the two cannot drift.  A term promoted from objective to
    barrier (or a new objective term) changes this tuple on its next import, and
    `Priorities` starts rejecting the old allocation by name — which is the correct
    failure, because an allocation over a set of `should`s that has changed is not the
    allocation anyone approved.
    """
    return tuple(_objective_module().OBJECTIVE_TERMS)


# ---------------------------------------------------------------------------
# CONSTANTS OF THE DERIVATION
# ---------------------------------------------------------------------------

G_MS2 = 9.80665                 # standard gravity, exact by definition (CGPM 1901)

# The stroke a field class asks for, in mm of axle drop at the service load.
#
# THIS IS THE ONLY THING `field_class` REACHES, and the table is a POLICY rather than a
# measurement — nothing in this tree has run a wheel over an obstacle, because the ground
# is flat, rigid and frictionless (see the module docstring).  It is stated as a table so
# it can be argued with, in the form `MARGIN_KNEE_UTIL = 0.80` already uses.
#
# `grass` IS 2.0 AND THAT IS NOT A COINCIDENCE: it is `TARGET_DEFLECTION_MM`, and the
# whole point of Step 0 is that this table has to reproduce the shipped constant exactly
# or the derivation is wrong.  The other two rows are the honest interpolation around it —
# a paved strip needs no travel it does not already have, a rough field needs more.
STROKE_BY_FIELD_CLASS = {"paved": 1.0, "grass": 2.0, "rough": 3.5}

# The floor under any stroke requirement, in mm.
#
# 1.0 IS AN IN-TREE CITATION AND NOT A ROUND NUMBER: `wheel_fea.py:156` records that
# `TARGET_DEFLECTION_MM` was *"raised from 1.0 for more travel"*, so 1.0 mm is the
# smallest stroke this repository has ever actually optimised a wheel to and the smallest
# it has any evidence it can hit.  Below it the target starts to sit inside the axle
# drop's own mesh-convergence band (M8b-i.5 measured the drop's GCI at 0.14%-2.44%, i.e.
# up to 0.05 mm at a 2 mm target), and a requirement finer than the instrument is not a
# requirement.
STROKE_FLOOR_MM = 1.0

# Fraction of `F * stroke` the wheel actually stores on the way down.
#
# 0.5 IS THE LINEAR-SPRING FIGURE and it is an assumption, named here rather than buried
# in the algebra: a spring that reaches force `F` at deflection `s` has stored `F*s/2`,
# not `F*s`.  A stiffening spring does better than 0.5 and a softening one worse, and
# this tree has never measured the wheel's force-deflection curve away from the design
# point — `axle_drop` is evaluated at ONE service force (`service_qoi_value_and_grad`), so
# the curve is a single point and its shape is not knowable from any committed artifact.
# Measuring it is a successor, and until then this is the conservative-in-the-right-
# direction choice: assuming MORE efficiency would shorten the implied stroke and inflate
# the implied sink rate, i.e. flatter the design.
STROKE_EFFICIENCY = 0.5

# ---------------------------------------------------------------------------
# SERVICE LIFE -> SAFETY FACTOR
# ---------------------------------------------------------------------------
# `wheel_fea.SAFETY_FACTOR = 1.6` is pure policy today with no stated life behind it.
# This does not replace the number, it says WHAT LIFE IT IS THE FACTOR FOR, so that
# asking for ten times the life becomes an arithmetic question instead of an argument.
#
# THE FORM IS A POLICY, MARKED AS ONE.  A decade of landings costs
# `FATIGUE_DECADE_SLOPE` of safety factor above the reference life, flat below it.  There
# is no S-N curve for this material, this print orientation and this load spectrum
# anywhere in this repository, and inventing one would be exactly the "fitted rule
# believed without a hold-out" that cost §73 half an arc.  What IS defensible is that the
# knockdown is monotone in life and roughly log-linear over a few decades, and that is all
# this claims.
#
# NOT TO BE CONFUSED WITH `studies/study_deflection_gci.py:72`'s `SAFETY_FACTOR = 1.25`,
# which is ROACHE'S GCI factor and has nothing to do with structural margin.  Two
# different `SAFETY_FACTOR`s live in this repo; a global rename would silently corrupt a
# convergence gate.
SAFETY_FACTOR_BASE = W.SAFETY_FACTOR       # 1.6, unchanged and read rather than retyped
REFERENCE_LANDINGS = 1000
FATIGUE_DECADE_SLOPE = 0.15


def fatigue_knockdown(landings):
    """Multiplier on the base safety factor for `landings` cycles.  1.0 at or below
    `REFERENCE_LANDINGS`, then `FATIGUE_DECADE_SLOPE` per decade above it.

    Monotone non-decreasing in `landings` by construction, which is the only property the
    rest of this module relies on: more life is never cheaper.
    """
    n = float(landings)
    if not n > 0:
        raise ValueError(f"landings must be positive, got {landings!r}")
    if n <= REFERENCE_LANDINGS:
        return 1.0
    return 1.0 + FATIGUE_DECADE_SLOPE * float(np.log10(n / REFERENCE_LANDINGS))


def safety_factor(landings):
    """`SF_base * k_fatigue(N)` — the divisor under the allowable stress."""
    return SAFETY_FACTOR_BASE * fatigue_knockdown(landings)


# ---------------------------------------------------------------------------
# THE MATERIAL CARD, AND THE TEMPERATURE MODEL
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaterialCard:
    """Everything about the material that a requirement can move, in one place.

    `e_retention_anchors` and `sigma_retention_anchors` are `((T_c, retention), ...)`
    ascending in `T_c`, both carrying an EXACT 1.0 at 20 C so that the baseline is
    untouched by construction rather than by luck.
    """

    name: str
    e_20c_mpa: float
    sigma_ult_20c_mpa: float
    fff_knockdown: float
    nu: float
    density_g_mm3: float
    t_max_service_c: float
    e_retention_anchors: tuple
    sigma_retention_anchors: tuple

    def __post_init__(self):
        for what, anchors in (("e", self.e_retention_anchors),
                              ("sigma", self.sigma_retention_anchors)):
            t = [a[0] for a in anchors]
            r = [a[1] for a in anchors]
            if list(t) != sorted(t) or len(set(t)) != len(t):
                raise ValueError(f"{self.name}: {what} anchors must ascend in T, got {t}")
            if any(b > a for a, b in zip(r, r[1:])):
                raise ValueError(
                    f"{self.name}: {what} retention must be monotone non-increasing in "
                    f"T — a material that gets stiffer as it heats is not what this "
                    f"knockdown models; got {r}")
            if (20.0, 1.0) not in [(float(a), float(b)) for a, b in anchors]:
                raise ValueError(
                    f"{self.name}: {what} anchors must contain (20.0, 1.0) EXACTLY, so "
                    f"that a 20 C requirement set reproduces the shipped constants bit "
                    f"for bit rather than to a tolerance")

    # -- the two curves -----------------------------------------------------

    def _retain(self, anchors, ambient_c, what):
        t = float(ambient_c)
        if t > self.t_max_service_c:
            raise ValueError(
                f"{self.name}: ambient {t} C is above t_max_service_c "
                f"{self.t_max_service_c} C.  REFUSED rather than extrapolated: above the "
                f"last anchor this material is on the wrong side of its glass transition "
                f"and a linear extrapolation of a collapsing modulus is not a knockdown, "
                f"it is a guess with a sign.")
        xs = np.array([a[0] for a in anchors], dtype=float)
        ys = np.array([a[1] for a in anchors], dtype=float)
        if t < xs[0]:
            raise ValueError(
                f"{self.name}: ambient {t} C is below the coldest {what} anchor "
                f"{xs[0]} C; this table is an interpolation and does not extrapolate.")
        return float(np.interp(t, xs, ys))

    def e_retention(self, ambient_c):
        """Young's modulus at `ambient_c` as a fraction of its 20 C value."""
        return self._retain(self.e_retention_anchors, ambient_c, "modulus")

    def sigma_retention(self, ambient_c):
        """Ultimate stress at `ambient_c` as a fraction of its 20 C value."""
        return self._retain(self.sigma_retention_anchors, ambient_c, "strength")

    def e_mpa(self, ambient_c):
        return self.e_20c_mpa * self.e_retention(ambient_c)

    def allowable_stress_mpa(self, ambient_c, landings):
        """`sigma_ult(20C) * retention(T) * fff_knockdown / SF(landings)`.

        The three knockdowns multiply and their ORDER IS NOT ARBITRARY as prose: the FFF
        factor is a property of how the part is made, the retention of where it is used,
        and the safety factor of how long — so the first two shrink the material and the
        third divides what is left.  At 20 C, `REFERENCE_LANDINGS` landings this is
        `50.0 * 1.0 * 0.80 / 1.6 = 25.0` = `wheel_fea.ALLOWABLE_STRESS_MPA`, exactly.
        """
        return (self.sigma_ult_20c_mpa * self.sigma_retention(ambient_c)
                * self.fff_knockdown / safety_factor(landings))


# THE THERMAL ANCHORS.  READ THE SCOPE NOTE BELOW BEFORE QUOTING ANY HOT-DAY RESULT.
#
# WHAT THIS IS: a QUASI-STATIC KNOCKDOWN, stated as a piecewise-linear INTERPOLATION
# BETWEEN NAMED ANCHOR POINTS.  It is NOT a model, and the distinction is the deliverable:
# nothing here is fitted, so nothing here can be validated by holding a point out.
# `studies/study_mbse_baseline.py` reports the hold-out error at every interior anchor
# anyway, as the honest statement of how much the table is asserting between its own
# points — and it is not small (7-9% at the knee), which is precisely why this is
# published as an interpolation and not as a curve.
#
# WHAT IT IS NOT, AND THIS SENTENCE TRAVELS WITH EVERY HOT PROFILE THIS REPO EVER PRINTS:
# **no creep, no fatigue, no thermal expansion, no self-heating, no rate dependence.**
# PLA creeps badly above ~45 C, and a static allowable at elevated temperature is
# OPTIMISTIC — a part that passes this check at 50 C may still sag under a week of
# standing load, and nothing here would notice.
#
# WHERE THE NUMBERS COME FROM, STATED PLAINLY BECAUSE THIS TREE REQUIRES IT: they are an
# ENGINEERING TABLE for a PLA part below and through its glass transition (Tg ~ 55-60 C),
# entered as the shape of the behaviour — flat-ish to 40 C, a knee through 50-55 C, and
# most of the stiffness gone by 60 C.  **They are not a measurement made in this tree and
# they are not traceable to a datasheet inside this repository.**  That is the same
# standing as `MARGIN_KNEE_UTIL`'s 0.80 ("a JUDGEMENT about a printed PLA part, not a
# measurement, and it is the number to argue with") and it is recorded with the same
# words.  A DMA sweep of the actual filament is the successor that retires this comment.
#
# `t_max_service_c = 60.0` is a HARD REFUSAL, not a clamp: above it the material is
# through Tg, the retention curve is falling off a cliff, and a linear extrapolation of a
# cliff is worse than no answer.
PLA_FFF = MaterialCard(
    name="PLA-FFF",
    e_20c_mpa=W.YOUNGS_MODULUS_PLA_MPA,            # 2300.0
    # `wheel_fea.ULTIMATE_STRESS_MPA` is ALREADY knocked down (50.0 * FFF_KNOCKDOWN); the
    # card carries the two factors SEPARATELY because temperature scales the material and
    # not the process.  50.0 is recovered rather than retyped so the two cannot drift.
    sigma_ult_20c_mpa=W.ULTIMATE_STRESS_MPA / W.FFF_KNOCKDOWN,
    fff_knockdown=W.FFF_KNOCKDOWN,                 # 0.80
    nu=0.35,                                       # `wheel_fem.POISSON_RATIO_PLA`
    density_g_mm3=W.DENSITY_PLA,
    t_max_service_c=60.0,
    e_retention_anchors=((-20.0, 1.18), (0.0, 1.09), (20.0, 1.0), (40.0, 0.85),
                         (50.0, 0.68), (55.0, 0.45), (60.0, 0.15)),
    sigma_retention_anchors=((-20.0, 1.12), (0.0, 1.06), (20.0, 1.0), (40.0, 0.86),
                            (50.0, 0.72), (55.0, 0.55), (60.0, 0.25)),
)

THERMAL_SCOPE_NOTE = (
    "QUASI-STATIC KNOCKDOWN ONLY — no creep, no fatigue, no thermal expansion, no "
    "self-heating, no rate dependence.  PLA creeps badly above ~45 C, so a static "
    "allowable at elevated temperature is OPTIMISTIC.  The retention curves are a "
    "piecewise-linear interpolation between asserted engineering anchors, not a fitted "
    "model and not a measurement made in this tree."
)


# ---------------------------------------------------------------------------
# THE MISSION
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Mission:
    """The absolute facts.  Every field reaches at least one constant, and the docstring
    of each derivation says which — and, where it matters more, which it does NOT.

    `k_asym` is the one field that is neither a vehicle property nor an environment: it
    is the share of the static load one wheel carries when it arrives first, and it is a
    policy number the way `SAFETY_FACTOR` is.  Named rather than folded into `n_wheels`
    because dividing by fewer wheels would ALSO change the static reaction, and those are
    two different statements.
    """

    auw_kg: float
    n_wheels: int
    k_asym: float
    sink_rate_ms: float
    field_class: str
    ambient_c: float
    landings: int
    nozzle_mm: float
    perimeters: int
    material: MaterialCard = PLA_FFF

    def __post_init__(self):
        if self.field_class not in STROKE_BY_FIELD_CLASS:
            raise ValueError(
                f"field_class {self.field_class!r} is not one of "
                f"{sorted(STROKE_BY_FIELD_CLASS)}")
        for name in ("auw_kg", "k_asym", "sink_rate_ms", "nozzle_mm"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")
        if int(self.n_wheels) < 1 or int(self.perimeters) < 1:
            raise ValueError("n_wheels and perimeters must be >= 1")

    # -- the load path ------------------------------------------------------

    @property
    def weight_n(self):
        return float(self.auw_kg) * G_MS2

    @property
    def static_force_n(self):
        """The reaction at one wheel with the vehicle standing still."""
        return self.weight_n / float(self.n_wheels)

    @property
    def stroke_mm(self):
        """The stroke requirement, and it is set BEFORE the load factor that needs it.

        THE CIRCULARITY IS REAL AND IT IS RESOLVED BY ORDERING, NOT BY A FIXED POINT.
        `n_land` depends on the stroke, the stroke is the deflection requirement, and the
        deflection requirement is what the optimiser is trying to hit — so a fixed-point
        stroke would make the LOAD a function of the DESIGN.  This repository loads to a
        FORCE and not to an indentation, and that distinction is the entire subject of
        `wheel_adjoint.service_qoi_value_and_grad`'s docstring (:649-662), where it is
        *"not a correction but the term"*.  So: field class and the floor set the stroke,
        and the load factor follows from it.  Do not solve the fixed point.
        """
        return max(STROKE_FLOOR_MM, STROKE_BY_FIELD_CLASS[self.field_class])

    @property
    def effective_stroke_m(self):
        return self.stroke_mm * STROKE_EFFICIENCY / 1000.0

    @property
    def landing_load_factor(self):
        """`n_land = 1 + v_z^2 / (2 g s_eff)` — the energy balance over the stroke.

        The `1 +` is the weight's own work over the stroke; the second term is the sink
        energy.  This is THE number MBSE_PLAN.md Step 0 exists to make visible, because a
        2.0 mm stroke is a very short stroke and the factor it implies is large.
        """
        return 1.0 + self.sink_rate_ms ** 2 / (2.0 * G_MS2 * self.effective_stroke_m)

    @property
    def force_n(self):
        """The service load on ONE wheel — `wheel_fea.TOTAL_FORCE_NEWTONS`' successor."""
        return self.static_force_n * self.landing_load_factor * float(self.k_asym)

    def sink_rate_for_force(self, force_n):
        """INVERT the load path: the sink rate that produces `force_n` at this vehicle.

        Step 0 runs the derivations backwards, and this is the half that has an
        analytic inverse — `n_land` is monotone in `v_z^2`, so there is exactly one
        non-negative root and no search is needed.
        """
        n_land = float(force_n) / (self.static_force_n * float(self.k_asym))
        if n_land < 1.0:
            raise ValueError(
                f"force {force_n} N is below this vehicle's own static reaction times "
                f"k_asym ({self.static_force_n * self.k_asym:.4f} N), so no sink rate "
                f"produces it — the load factor would have to be less than 1.")
        return float(np.sqrt((n_land - 1.0) * 2.0 * G_MS2 * self.effective_stroke_m))

    # -- everything else ----------------------------------------------------

    @property
    def min_wall_mm(self):
        """`nozzle_mm * perimeters`.  The highest-leverage line in this file.

        `MIN_WALL_MM` SETS 4 OF THE 14 GENES AT THE OPTIMUM — the shipped genome has
        `t1 = t2 = 1.2` exactly on the floor — so this is one of the most load-bearing
        numbers in the tree, and until now it was derived in a PROSE COMMENT
        (`# 3 perimeters @ 0.4 mm nozzle`) and written as a literal.  Nothing in code
        knew 1.2 came from 0.4.  Now something does.
        """
        return float(self.nozzle_mm) * int(self.perimeters)

    @property
    def safety_factor(self):
        return safety_factor(self.landings)

    @classmethod
    def implied_baseline(cls):
        """THE MISSION THE SHIPPED CONSTANTS IMPLY — MBSE_PLAN.md Step 0's answer.

        Every field but one is a stated choice about a plausible small UAV, and the last
        one is SOLVED so that `Requirements.from_mission` of this reproduces
        `wheel_fea`'s four constants.  `sink_rate_ms` is the solved field, and it is
        computed here through `sink_rate_for_force` rather than pasted in as a literal,
        so the round trip is exact by construction and nobody has to trust a transcribed
        float.

        **THE VEHICLE IS A FIAT AND IT IS SAID SO HERE.**  `FORCE_LBS = 15.0` is one
        equation in four unknowns (`auw_kg`, `n_wheels`, `k_asym`, `sink_rate_ms`), so
        there is a three-parameter FAMILY of missions that reproduce 66.7233 N and no
        way to pick between them from anything on disk.  3 kg on three wheels with a 1.5
        asymmetry factor is chosen because the part itself argues for it: the shipped
        wheel is 48.64 g as an OCC solid, three of them are 146 g, and a landing gear
        that is ~5% of all-up weight puts the vehicle at a few kg.
        `studies/study_mbse_baseline.py` prints the rest of the family so the fiat can be
        checked rather than believed.

        **AND THE SOLVED SINK RATE IS THE FINDING, NOT THE FIAT.**  Across the whole
        plausible family it comes out in the low tenths of a m/s — a taxi-speed arrival,
        not a landing — and that is a property of the 2.0 mm STROKE, which every member
        of the family shares.  See the driver.
        """
        m = cls(auw_kg=3.0, n_wheels=3, k_asym=1.5, sink_rate_ms=0.0,
                field_class="grass", ambient_c=20.0, landings=REFERENCE_LANDINGS,
                nozzle_mm=0.4, perimeters=3)
        return replace(m, sink_rate_ms=m.sink_rate_for_force(W.TOTAL_FORCE_NEWTONS))

    def as_dict(self):
        d = {k: getattr(self, k) for k in
             ("auw_kg", "n_wheels", "k_asym", "sink_rate_ms", "field_class",
              "ambient_c", "landings", "nozzle_mm", "perimeters")}
        d["material"] = self.material.name
        return d


# ---------------------------------------------------------------------------
# THE PRIORITIES
# ---------------------------------------------------------------------------
#
# THE REFERENCE DEVIATION `d_T` — one named physical unit of MISSING a requirement — and
# `c_T = L(d_T)`, the loss that miss costs at the reference genome under DEFAULT_WEIGHTS.
#
# WHY `L(d_T)` AND NOT `dL/dx` AT THE CURRENT ITERATE, AND NOT LOSS SHARE.  At the shipped
# genome the five objective terms split the loss 0.013% / 99.07% / 0.40% / 0.51% / 0.00%,
# which reads as "this tree cares 99% about mass and 0.013% about stroke" and is FALSE:
# `deflection` is small because the design sits at 1.99742 mm against a 2.0 mm target — a
# -0.129% miss — so the term is small because the requirement is MET.  **Loss share
# measures satisfaction, not priority.**  The marginal rate fails for the same reason from
# the other side: at a satisfied quadratic term it is near zero, so a calibration built on
# it would be a calibration built on where the last descent happened to stop.  `L(d_T)` is
# a property of the WEIGHT, not of the iterate.
#
# `smoothness`'s deviation is relative because its argument is an integral with no
# reference scale in the table — the term is LINEAR in that integral
# (`wheel_objective.py:823-826`), so 1% of it costs 1% of the term and the reference
# genome supplies the scale.  That is the one `c_T` that must be MEASURED off a committed
# artifact rather than read off the weight table, and `studies/study_mbse_calibration.py`
# is where it is measured.
REFERENCE_DEVIATION = {
    "mass": ("1% of MASS_REFERENCE_G", 0.01 * W.MASS_REFERENCE_G),
    "deflection": ("1% relative error on the stroke target", 0.01),
    "stress_margin": ("1% of utilisation above MARGIN_KNEE_UTIL", 0.01),
    "smoothness": ("1% of the curvature-rate integral at the reference genome", 0.01),
    "phase_ripple": ("1 point of std/mean axle drop", 0.01),
}

# `phase_ripple`'s reference deviation is the one number in the table above that could not
# be read off an existing weight, because `DEFAULT_WEIGHTS["phase_ripple"] = 0.0` — the
# term is switched off and always has been ("off by default; gate 10 reports what turning
# it on costs").  So it is stated, at the same standing as `MARGIN_KNEE_UTIL`'s 0.80: one
# POINT of ripple, 0.01 of std/mean, matched to `stress_margin`'s "1% of utilisation"
# because both are absolute fractions of a dimensionless ratio.
#
# AND THE SHIPPED WHEEL IS 0.1044286 — ten and a half points — so this reference is a
# small deviation against a large standing one.  That number is carried beside this
# constant on purpose: it is what makes the anchor arguable instead of decorative, and
# `studies/study_mbse_calibration.py` prices exactly what buying `rolling` would cost.
RIPPLE_REFERENCE_DEVIATION = REFERENCE_DEVIATION["phase_ripple"][1]


@dataclass(frozen=True)
class Priorities:
    """A 100-point zero-sum allocation over `priority_axes()`.

    **THE BUDGET IS NOT A UI CONVENTION, IT IS A CONSERVATION LAW.**  Weights are not
    scale-free in this objective: `BARRIER_TERMS` are absolute, so multiplying every
    objective weight by two does not leave the optimum alone — it HALVES the effective
    strength of every `shall`.  `sum p = 100` is precisely what forbids a user from
    buying more of everything and quietly weakening every feasibility barrier in the
    process.  `weights_from_priorities` preserves it and
    `tests/test_requirements.py` gates it.
    """

    points: dict

    TOLERANCE = 1e-9

    def __post_init__(self):
        axes = priority_axes()
        got = tuple(self.points)
        unknown = [k for k in got if k not in axes]
        missing = [k for k in axes if k not in got]
        if unknown or missing:
            raise ValueError(
                f"a priority allocation is over exactly the {len(axes)} objective terms "
                f"{axes} — unexpected {sorted(unknown)}, missing {missing}.  Points may "
                f"not reach a BARRIER term: a barrier is a `shall` whose only admissible "
                f"value is zero and it is never traded against.")
        neg = {k: v for k, v in self.points.items() if float(v) < 0.0}
        if neg:
            raise ValueError(f"points must be non-negative, got {neg}")
        total = float(sum(float(v) for v in self.points.values()))
        if abs(total - 100.0) > self.TOLERANCE:
            raise ValueError(
                f"a priority allocation must sum to 100, got {total!r}.  The budget is "
                f"the conservation law that keeps the objective-against-barrier balance "
                f"fixed while priorities move — a budget that does not bind is not a "
                f"budget.")
        # Frozen dataclass: normalise the mapping in place so callers cannot mutate it
        # out from under a `req_hash`.
        object.__setattr__(self, "points",
                           {k: float(self.points[k]) for k in axes})

    def as_dict(self):
        return dict(self.points)


def reference_costs(smoothness_loss, weights=None):
    """`c_T` for all five objective terms, from the WEIGHT TABLE and one measured scale.

    `smoothness_loss` is that term's value at the reference genome UNDER
    `DEFAULT_WEIGHTS`, and it is the only input that cannot come from the weight table:
    `smoothness`' argument is an integral with no reference scale of its own, so the
    genome supplies the scale.  It is divided back out by `DEFAULT_WEIGHTS["smoothness"]`
    here rather than used raw, so that `c_smoothness` is correct at ANY weights and not
    only at the default — which is what lets `weights_from_priorities` be checked for
    conservation by feeding its own output straight back in.  Everything else is closed
    form:

        mass          w_mass * d                    (LINEAR in grams)
        deflection    w_deflection * d^2            (quadratic in relative error)
        stress_margin w_margin * d^2                (`soft_barrier` above the knee)
        smoothness    w_smooth * integral * d       (LINEAR in the integral)
        phase_ripple  w_ripple * d^2                (quadratic; w is 0.0 by default)

    Returns a dict keyed by `priority_axes()`.
    """
    WO = _objective_module()
    base = dict(WO.DEFAULT_WEIGHTS)
    w = base if weights is None else dict(weights)
    d = {k: v[1] for k, v in REFERENCE_DEVIATION.items()}
    integral = float(smoothness_loss) / base["smoothness"]
    return {
        "deflection": w["deflection"] * d["deflection"] ** 2,
        "mass": w["mass"] * d["mass"],
        "stress_margin": w["stress_margin"] * d["stress_margin"] ** 2,
        "smoothness": w["smoothness"] * integral * d["smoothness"],
        "phase_ripple": w["phase_ripple"] * d["phase_ripple"] ** 2,
    }


def calibrated_priorities(smoothness_loss, weights=None):
    """`p_cal_T = 100 * c_T / sum(c)` — WHAT THE SHIPPED WEIGHT TABLE ALREADY ALLOCATES.

    Not a proposal.  This is the portfolio `DEFAULT_WEIGHTS` has been running since it was
    written, stated in points for the first time, and it is nothing like the 99%-mass
    reading its loss breakdown invites.

    Returns `(Priorities, costs)` — the allocation and the `c_T` it came from, because
    every caller that wants one wants the other beside it.
    """
    c = reference_costs(smoothness_loss, weights)
    total = sum(c.values())
    if not total > 0.0:
        raise ValueError("every objective term has zero reference cost; nothing to allocate")
    axes = priority_axes()
    p = {k: 100.0 * c[k] / total for k in axes}
    # Land the rounding on ONE axis rather than letting `sum p` miss 100 by an ulp and
    # trip the budget check.  The LARGEST share carries it, so the correction is the
    # smallest relative perturbation available — it is at most a few ulp, which
    # `weights_from_priorities`' identity test is what proves.
    p[max(axes, key=lambda k: p[k])] += 100.0 - sum(p.values())
    return Priorities(p), c


def cost_per_point(smoothness_loss, weights=None):
    """`sum(c_cal) / 100` — the exchange-rate pressure ONE point buys, in loss units.

    THE WHOLE MAP IS THIS NUMBER.  `weights_from_priorities` sets each term's weight so
    that its own reference deviation costs `cost_per_point * p_T`, which makes total
    pressure `sum_T c_T(p) = cost_per_point * sum_T p_T` — invariant exactly when the
    points sum to 100, for every allocation, with no term singled out.
    """
    return sum(reference_costs(smoothness_loss, weights).values()) / 100.0


def weights_from_priorities(priorities, smoothness_loss, weights=None):
    """A 14-key weight dict: the five objective terms re-priced, the barriers untouched.

    THE RULE, IN ONE SENTENCE: `w_T(p)` is whatever weight makes term `T`'s own reference
    deviation cost `cost_per_point * p_T`.

    For the four terms whose default weight is nonzero this is identical to the obvious
    map `w_T = DEFAULT_WEIGHTS[T] * p_T / p_cal_T`, and at `p = p_cal` it returns
    `DEFAULT_WEIGHTS` exactly — the map is an IDENTITY at its own calibration point, or
    it is not a re-parameterisation but a change.

    **AND IT IS DEFINED ON `phase_ripple`, WHICH THE OBVIOUS MAP IS NOT.**  That term's
    default weight is 0.0, so `c = 0`, so `p_cal = 0`, so `w = w_default * p / p_cal` is
    `0/0` — undefined on the one axis a user is most likely to want to move first,
    because it is the one this tree has never bought any of.  Stated as cost-per-point
    the singularity is not there: `w_ripple(p) = cost_per_point * p / d_ripple^2`, which
    is 0.0 at `p = 0` and finite everywhere else.  The same treatment applies to any
    future objective term whose default weight is zero, with no special case in this
    function.

    `smoothness_loss` is the reference genome's smoothness term — see `reference_costs`.
    """
    WO = _objective_module()
    base = dict(WO.DEFAULT_WEIGHTS if weights is None else weights)
    if not isinstance(priorities, Priorities):
        priorities = Priorities(priorities)
    unit = cost_per_point(smoothness_loss, weights)
    d = {k: v[1] for k, v in REFERENCE_DEVIATION.items()}

    out = dict(base)
    for term, p in priorities.points.items():
        target = unit * p                      # the cost this term's reference miss owes
        if term == "mass":
            out[term] = target / d["mass"]
        elif term == "smoothness":
            # LINEAR in the integral, whose scale is the reference genome's own term
            # value divided by the weight it was measured under — the same quantity
            # `reference_costs` calls `integral`, computed the same way so the two are
            # exact inverses.
            integral = float(smoothness_loss) / _default_weights()["smoothness"]
            out[term] = target / (integral * d["smoothness"])
        else:                                   # quadratic terms
            out[term] = target / d[term] ** 2
    return out


# ---------------------------------------------------------------------------
# THE REQUIREMENT SET
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Requirements:
    """Everything the objective needs from the mission, in the objective's own units.

    Six numbers and a weight table.  `objective(genes, req=...)` routes exactly these,
    and MBSE_PLAN.md's load-bearing check is that `req=Requirements.baseline()` is
    bit-identical to passing nothing at all.
    """

    force_n: float
    target_deflection_mm: float
    allowable_stress_mpa: float
    min_wall_mm: float
    e_mpa: float
    nu: float
    weights: dict
    provenance: dict = field(default_factory=dict)

    # -- construction -------------------------------------------------------

    @classmethod
    def baseline(cls):
        """TODAY'S CONSTANTS, READ FROM THE MODULES THAT DEFINE THEM.

        NOT derived from a mission, and that is deliberate: MBSE_PLAN.md's "do not move a
        single default" is enforceable only if the baseline is the constants themselves
        rather than a derivation that happens to land on them.  `Mission.implied_baseline`
        is the derivation, and Step 0's job is to show the two agree — a claim that means
        something precisely because they are computed two different ways.
        """
        WO = _objective_module()
        return cls(
            force_n=float(W.TOTAL_FORCE_NEWTONS),
            target_deflection_mm=float(W.TARGET_DEFLECTION_MM),
            allowable_stress_mpa=float(W.ALLOWABLE_STRESS_MPA),
            min_wall_mm=float(W.MIN_WALL_MM),
            e_mpa=float(W.YOUNGS_MODULUS_PLA_MPA),
            nu=float(PLA_FFF.nu),
            weights=dict(WO.DEFAULT_WEIGHTS),
            provenance={"source": "wheel_fea + wheel_objective module constants",
                        "derived": False},
        )

    @classmethod
    def from_mission(cls, mission, priorities=None, smoothness_loss=None):
        """Run the derivations forward.

        `priorities` and `smoothness_loss` travel together: re-pricing the weights needs
        the reference genome's `smoothness` term (see `reference_costs`), so asking for
        one without the other is refused rather than defaulted — a silently assumed
        reference scale is a calibration nobody can audit.
        """
        WO = _objective_module()
        if (priorities is None) != (smoothness_loss is None):
            raise ValueError(
                "priorities and smoothness_loss must be given together: the points -> "
                "weights map is anchored at a reference genome and there is no default "
                "reference to guess.")
        weights = (dict(WO.DEFAULT_WEIGHTS) if priorities is None
                   else weights_from_priorities(priorities, smoothness_loss))
        mat = mission.material
        return cls(
            force_n=mission.force_n,
            target_deflection_mm=mission.stroke_mm,
            allowable_stress_mpa=mat.allowable_stress_mpa(mission.ambient_c,
                                                          mission.landings),
            min_wall_mm=mission.min_wall_mm,
            e_mpa=mat.e_mpa(mission.ambient_c),
            nu=mat.nu,
            weights=weights,
            provenance={
                "source": "wheel_requirements.Requirements.from_mission",
                "derived": True,
                "mission": mission.as_dict(),
                "priorities": None if priorities is None else priorities.as_dict(),
                "weight_n": mission.weight_n,
                "static_force_n": mission.static_force_n,
                "stroke_mm": mission.stroke_mm,
                "effective_stroke_m": mission.effective_stroke_m,
                "landing_load_factor": mission.landing_load_factor,
                "safety_factor": mission.safety_factor,
                "fatigue_knockdown": fatigue_knockdown(mission.landings),
                "e_retention": mat.e_retention(mission.ambient_c),
                "sigma_retention": mat.sigma_retention(mission.ambient_c),
                "thermal_scope": THERMAL_SCOPE_NOTE,
            },
        )

    # -- consumption --------------------------------------------------------

    def objective_kwargs(self):
        """The keyword dict `wheel_objective.objective` takes.

        `min_wall_mm` IS NOT IN IT, and the omission is the point: the wall floor is not
        a term in the loss, it is the LOW BOUND on four genes, and it reaches the search
        through `wheel_fea.set_min_wall` — which rewrites `GENE_SPACE` and re-snapshots
        the three arrays the GA clips against.  `apply_process()` is that call.  Passing
        it here would put it somewhere it does nothing and read as though it had.
        """
        return {"force": self.force_n,
                "target_deflection_mm": self.target_deflection_mm,
                "allowable_stress_mpa": self.allowable_stress_mpa,
                "weights": dict(self.weights),
                "E": self.e_mpa, "nu": self.nu}

    def apply_process(self):
        """Move `wheel_fea.MIN_WALL_MM` and the gene box with it.  Returns the old floor.

        MUTATES MODULE STATE, on purpose and in one place, because that is the only way
        the floor can move inside one interpreter at all — `GENE_SPACE` consumes
        `MIN_WALL_MM` at IMPORT time (`wheel_fea.py:259-262`).  Call it BEFORE
        `bounds_arrays`, exactly as `wheel_stage3.main` already does for `--min-wall`.
        """
        old = float(W.MIN_WALL_MM)
        if self.min_wall_mm != old:
            W.set_min_wall(self.min_wall_mm)
        return old

    def as_dict(self):
        """The record's top-level `requirements` block.  TOP-LEVEL, never inside `genes`:
        `wheel_genome.save_record` refuses a key there and states why — it *"changes
        `genome_hash` for every genome and breaks all staleness checks"*.
        """
        return {"force_n": self.force_n,
                "target_deflection_mm": self.target_deflection_mm,
                "allowable_stress_mpa": self.allowable_stress_mpa,
                "min_wall_mm": self.min_wall_mm,
                "e_mpa": self.e_mpa,
                "nu": self.nu,
                "weights": dict(self.weights),
                "req_hash": self.req_hash(),
                "provenance": dict(self.provenance)}

    def req_hash(self):
        """Short stable fingerprint of the six numbers and the weight table.

        Same construction and same 7-character width as `wheel_genome.genome_hash`, and
        deliberately NOT over `provenance`: two missions that derive the same constants
        are the same requirement for the optimiser, and a record scored under one must
        not be refused against the other.  What the hash protects is the
        `warn_if_stale` discipline `wheel_step_export.py:196` already applies to STEP
        files — a record compared against requirements it was not descended under.
        """
        canon = json.dumps({
            "force_n": round(float(self.force_n), 12),
            "target_deflection_mm": round(float(self.target_deflection_mm), 12),
            "allowable_stress_mpa": round(float(self.allowable_stress_mpa), 12),
            "min_wall_mm": round(float(self.min_wall_mm), 12),
            "e_mpa": round(float(self.e_mpa), 12),
            "nu": round(float(self.nu), 12),
            "weights": {k: round(float(v), 12) for k, v in sorted(self.weights.items())},
        }, sort_keys=True)
        return hashlib.sha256(canon.encode()).hexdigest()[:7]

    # -- serialisation ------------------------------------------------------

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(self.as_dict(), fh, indent=2)
        return self.req_hash()


def load(path):
    """Read a requirements file written by `Requirements.save`."""
    with open(path) as fh:
        d = json.load(fh)
    req = Requirements(
        force_n=float(d["force_n"]),
        target_deflection_mm=float(d["target_deflection_mm"]),
        allowable_stress_mpa=float(d["allowable_stress_mpa"]),
        min_wall_mm=float(d["min_wall_mm"]),
        e_mpa=float(d["e_mpa"]), nu=float(d["nu"]),
        weights={k: float(v) for k, v in d["weights"].items()},
        provenance=dict(d.get("provenance", {})))
    stated = d.get("req_hash")
    if stated is not None and stated != req.req_hash():
        raise ValueError(
            f"{path} states req_hash {stated} but its own contents hash to "
            f"{req.req_hash()} — the file was edited without re-deriving it, so nothing "
            f"downstream can trust the hash to mean what it says.")
    return req


# ---------------------------------------------------------------------------
# VERIFICATION
# ---------------------------------------------------------------------------
#
# THE COMPLIANCE TABLE'S TWO HALVES ARE `wheel_objective`'s OWN, READ AND NOT RETYPED.
# `BARRIER_TERMS` are the `shall`s and `OBJECTIVE_TERMS` are the `should`s — that split
# is already in the code, already asserted disjoint and exhaustive
# (`wheel_objective.py:394-401`), and already load-bearing: it exists because defect 6
# promoted an infeasible design on 2026-08-11 by selecting on loss alone.  Reusing it is
# what stops this file inventing a second requirements taxonomy that can drift from the
# one the optimiser actually enforces.
#
# **A `should` CAN NEVER MAKE A DESIGN NON-COMPLIANT.**  That is the whole distinction: a
# barrier answers "may this design ship" and an objective answers "how good is it".  A
# missed `should` is reported with its margin and changes no verdict.

# Where a `shall` has a MEASURED quantity in a record's `metrics` block, so the table can
# show the physical number and not only the penalty it produced.  `(key, limit, sense)`;
# `sense` is the direction of compliance.  The six barriers not listed are geometric
# feasibility terms with no scalar metric of their own — their evidence IS the term value.
BARRIER_EVIDENCE = {
    "stress": ("stress_utilisation", 1.0, "<="),
    "buckling": ("buckling_ratio", 1.0, "<="),
    "min_sj": ("min_scaled_jacobian", None, ">="),      # limit read from wheel_objective
}

# Where a `should` has a stated reference point.  THREE OF THE FIVE DO NOT, and the table
# says so rather than inventing a budget.  `smoothness` and `phase_ripple` have no
# not-to-exceed anywhere in this tree.  `mass` LOOKED like it had one —
# `MASS_REFERENCE_G` — until PLAN.md §98 checked it: `Mission.implied_baseline`'s own
# docstring derives "~5% of all-up weight" from the shipped wheel's 48.64 g OCC-SOLID
# mass, and `verify()` here reads `metrics.mesh_mass_g`, the MESHED-AREA-x-width-x-density
# figure — 39.55 to 43.41 g, 11-19% under the solid.  THAT PAIR IS ONE DESIGN ON TWO
# MESHES, NOT TWO DESIGNS.  `FILLET_PLAN.md:3750` measures the same two numbers as the
# fillet's own mass — "9.8% heavier (39.548 g against 43.413 g; the fillet is material the
# unfilleted mesh" — and §105 reproduced them at `smoke` on the shipped genome: 39.5488
# unfilleted against 43.4161 filleted, +3.867 g, which is 10.6x this table's own 0.365 g
# tolerance for `mass`.  The conclusion below does not depend on it (neither mesh has a
# stated budget), but the spread is evidence about the MESH and must not be re-quoted as a
# design-to-design range.  The only stated precedent for a mass budget is denominated in a
# mass this function never sees, and reconciling the two conventions is unmeasured work, not
# arithmetic.  So `mass` joins the other two: reporting "NO LIMIT STATED" is more useful
# than a number nobody chose under the units this table actually checks, and it names the
# gap instead of hiding it behind a normaliser that was never a budget.
DEFLECTION_TOLERANCE = 0.05     # +/-5% of the stroke target, and it is a POLICY: the term
                                # is two-sided about the target because for a compliant
                                # mechanism the travel IS the feature, so a wheel that is
                                # too stiff misses the requirement exactly as a wheel that
                                # is too soft does.  5% is the band the deflection weight
                                # was itself calibrated against (`wheel_fea.py:158-163`,
                                # "a 5 % deflection error costs 2500*0.05^2 = 6.25").


def score_record(genes, req, cfg="coarse", *, n_phase=8, scheme="uniform",
                 kinematics="svk", **objective_kw):
    """One forward `objective` evaluation, reshaped into the record `verify` reads.

    Lives here rather than in a driver because BOTH the study driver and the `make mbse`
    front end need exactly this shape, and two copies of "which keys go in `metrics`"
    is how a compliance table and the artifact it claims to describe drift apart.

    `kinematics` DEFAULTS TO `svk` HERE AND NOWHERE ELSE IN THE TREE.  `wheel_fem`'s
    kernel default is `linear` on purpose (§32) and eleven study drivers never mention
    the argument at all, so a ladder built on those takes linear silently.  The shipped
    genome was descended under SVK, and a loss from one strain measure is not comparable
    with a loss from the other (§14).  A verifier that quietly answered a different
    question from the optimiser would be the same defect one layer up, so this one
    chooses, and says so in the record it returns.
    """
    import numpy as _np
    WO = _objective_module()
    phases = WO.phase_stencil(n_phase=n_phase, scheme=scheme)
    val, grad, brk = WO.objective(_np.asarray(genes, dtype=float), cfg, phases=phases,
                                  req=req, kinematics=kinematics, **objective_kw)
    return {
        "loss": float(val),
        "grad": [float(x) for x in _np.asarray(grad)],
        "loss_terms": {k: brk["terms"][k]["value"] for k in WO.TERMS},
        "metrics": brk["report"],
        "requirements": req.as_dict(),
        "scored_at": {"config": cfg, "kinematics": kinematics, "n_phase": n_phase,
                      "phase_scheme": scheme},
    }


def verify(record, req, strict=True):
    """A compliance table for one scored design against one requirement set.

    `record` is anything with `loss_terms` and `metrics` blocks — a genome record on
    disk, or a fresh `objective` breakdown reshaped into one.  `req` is the requirement
    set it was scored under.

    **A RECORD SCORED UNDER DIFFERENT REQUIREMENTS IS REFUSED, NOT SILENTLY COMPARED.**
    `wheel_step_export.warn_if_stale` (:196) already applies that discipline to STEP
    files; this is the same rule for requirements, and it matters more, because a stale
    STEP file looks wrong and a stale utilisation looks like a number.  A record carrying
    no `requirements` block at all is not refused — every artifact committed before this
    arc is such a record — but its provenance row says `unstated` and the caller has been
    told.

    Returns `{"rows": [...], "compliant": bool, "provenance": str}`; every row carries
    `id`, `kind` (`shall`/`should`), `statement`, `method`, `quantity`, `value`, `limit`,
    `margin` and `verdict`.
    """
    WO = _objective_module()
    stated = (record.get("requirements") or {}).get("req_hash")
    if stated is None:
        provenance = "unstated (record predates the requirements block)"
    elif stated != req.req_hash():
        if strict:
            raise ValueError(
                f"this record was scored under requirements {stated} and is being "
                f"verified against {req.req_hash()}.  REFUSED rather than compared: a "
                f"utilisation is a fraction of an allowable and an axle drop is measured "
                f"against a target, so two requirement sets produce numbers that look "
                f"alike and mean different things.  Re-score the genome, or pass "
                f"strict=False and read the table as a cross-comparison.")
        provenance = f"MISMATCH: scored under {stated}, verified against {req.req_hash()}"
    else:
        provenance = f"matched ({stated})"

    lt = record["loss_terms"]
    met = record.get("metrics", {})
    rows = []

    for term in WO.BARRIER_TERMS:
        value = float(lt[term])
        row = {"id": "SHALL-%s" % term.upper().replace("_", "-"), "kind": "shall",
               "statement": "the `%s` barrier is not breached" % term,
               "method": "analysis — Stage-3 objective barrier",
               "quantity": "loss_terms.%s" % term, "value": value,
               "limit": 0.0, "margin": -value,
               "verdict": "PASS" if value == 0.0 else "FAIL"}
        ev = BARRIER_EVIDENCE.get(term)
        if ev is not None:
            key, limit, sense = ev
            limit = WO.MIN_SJ_TARGET if limit is None else limit
            if key in met:
                row["evidence"] = {"quantity": "metrics.%s" % key,
                                   "value": float(met[key]), "limit": limit,
                                   "sense": sense}
        rows.append(row)

    for term in WO.OBJECTIVE_TERMS:
        rows.append(_should_row(term, lt, met, req, WO))

    return {"rows": rows,
            "compliant": all(r["verdict"] == "PASS"
                             for r in rows if r["kind"] == "shall"),
            "provenance": provenance,
            "req_hash": req.req_hash()}


def _should_row(term, lt, met, req, WO):
    """One `should` row.  Verdict is `MET`, `MISSED` or `NO LIMIT STATED`, and none of
    the three ever changes compliance."""
    base = {"id": "SHOULD-%s" % term.upper().replace("_", "-"), "kind": "should",
            "term_loss": float(lt[term])}
    if term == "deflection":
        v = float(met["axle_drop_mean_mm"])
        limit = req.target_deflection_mm
        rel = (v - limit) / limit
        base.update({
            "statement": "mean axle drop is within +/-%.0f%% of the stroke target"
                         % (100 * DEFLECTION_TOLERANCE),
            "method": "analysis — full-wheel contact FEA, mean over the phase stencil",
            "quantity": "metrics.axle_drop_mean_mm", "value": v, "limit": limit,
            "margin": rel,
            "verdict": "MET" if abs(rel) <= DEFLECTION_TOLERANCE else "MISSED"})
    elif term == "stress_margin":
        v = float(met["stress_utilisation"])
        limit = WO.MARGIN_KNEE_UTIL
        base.update({
            "statement": "worst junction utilisation stays under the margin knee",
            "method": "analysis — Kt * p-norm nominal / allowable",
            "quantity": "metrics.stress_utilisation", "value": v, "limit": limit,
            "margin": limit - v, "verdict": "MET" if v <= limit else "MISSED"})
    elif term == "mass":
        base.update({
            "statement": "no mass requirement exists in this tree (PLAN.md §98): the "
                         "one stated precedent, ~5% of all-up weight, is denominated in "
                         "OCC-solid mass and this row reads meshed mass instead",
            "method": "analysis — meshed area x width x density",
            "quantity": "metrics.mesh_mass_g", "value": float(met["mesh_mass_g"]),
            "limit": None, "margin": None, "verdict": "NO LIMIT STATED"})
    elif term == "phase_ripple":
        v = float(met["phase_ripple_std_over_mean"])
        base.update({
            "statement": "axle drop is uniform through one sector",
            "method": "analysis — std/mean over the phase stencil",
            "quantity": "metrics.phase_ripple_std_over_mean", "value": v,
            "limit": None, "margin": None, "verdict": "NO LIMIT STATED"})
    else:                                                       # smoothness
        base.update({
            "statement": "the centerline is a clean single-curvature spiral",
            "method": "analysis — curvature-rate integral plus reversal penalty",
            "quantity": "loss_terms.smoothness", "value": float(lt[term]),
            "limit": None, "margin": None, "verdict": "NO LIMIT STATED"})
    return base
