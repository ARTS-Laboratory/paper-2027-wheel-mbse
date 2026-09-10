"""The MBSE surface: a mission and 100 points in; a requirement set and a table out.

THIS MODULE ADDS NO MODELLING OF ITS OWN.  Every number below is computed by
`wheel_requirements` and every derivation is one of its properties; what is here is the
shaping into JSON and the three refusals a UI has to handle that a CLI does not.

WHY IT DOES NOT READ `studies/study_mbse_*.json`.  Those artifacts are anchored at genome
`09e8188` and carry a `stress_margin` weight of 325.0; the live `DEFAULT_WEIGHTS` value is
89.21 and the shipped genome moved to `b729e86` on 2026-09-06.  PLAN.md 119 declined to
refresh them on purpose -- *"refreshing one on a new genome does not update a stale
number; it rewrites the evidence under a conclusion the tree still makes"* -- so they are
correct as a record and wrong as a source.  Everything here calls the module.

WHY `c_smoothness` IS RE-READ ON EVERY DERIVE.  Four of the five reference costs come from
the weight table and are genome-independent.  The fifth cannot: `smoothness`' argument is
an integral with no reference scale, so the reference genome supplies it -- which means
the points->weights map RE-DERIVES AT EVERY PROMOTION.  Caching it across one would make
the panel quietly describe the previous wheel.

WHY `verify` IS CALLED TWICE.  It REFUSES, under `strict=True`, to compare a record scored
under one requirement set against another -- a utilisation is a fraction of an allowable
and an axle drop is measured against a target, so two requirement sets produce numbers
that look alike and mean different things.  A UI cannot simply propagate that as an error,
because cross-comparison is exactly what someone moving a slider is asking for.  So: try
strict, and on refusal fall back to `strict=False` and SAY SO in the payload.  A record
with no `requirements` block at all -- which `best_solution.json` is -- does not raise;
its provenance reads `unstated`, and that is shown rather than smoothed over.
"""

import json
import os
import sys
from dataclasses import replace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import wheel_fea as W                                             # noqa: E402
import wheel_genome as wg                                         # noqa: E402
import wheel_mbse as MB                                           # noqa: E402
import wheel_requirements as R                                    # noqa: E402
import wheel_objective as WO                                      # noqa: E402
import project_paths as PP                                        # noqa: E402

# The mission form. Order is the load path: what the vehicle is, then how it arrives,
# then how long it must last, then how it is printed.
MISSION_FIELDS = [
    ("auw_kg", "float", "kg", "All-up weight of the vehicle."),
    ("n_wheels", "int", "", "Wheels sharing the static load."),
    ("k_asym", "float", "", "Share of the static load one wheel carries arriving first. "
                            "A policy number, like the safety factor."),
    ("sink_rate_ms", "float", "m/s", "Arrival sink rate. The one field the implied "
                                     "baseline SOLVES rather than states."),
    ("field_class", "choice", "", "Surface: sets the stroke the wheel must give."),
    ("ambient_c", "float", "C", "Service temperature. Knocks down modulus and allowable "
                                "through the material card; above t_max_service_c the "
                                "card REFUSES rather than clamping."),
    ("landings", "int", "", "Service life in landings; drives the fatigue knockdown."),
    ("nozzle_mm", "float", "mm", "Printer nozzle diameter."),
    ("perimeters", "int", "", "Perimeter count. With the nozzle this sets the wall floor."),
]

# The derivations, in the order the load actually travels. Each is a Mission property.
DERIVATIONS = [
    ("weight_n", "N", "auw_kg x g"),
    ("static_force_n", "N", "weight / n_wheels -- standing still"),
    ("landing_load_factor", "", "arrival over static"),
    ("force_n", "N", "the service force the objective loads to"),
    ("stroke_mm", "mm", "the deflection requirement, set by field_class"),
    ("min_wall_mm", "mm", "nozzle x perimeters -- the printable floor on t0..t3"),
    ("safety_factor", "", "base factor with the fatigue knockdown for `landings`"),
]


def _reference_genome():
    """Whatever `best_solution.json` is RIGHT NOW. Never cached across a request."""
    return PP.BEST_SOLUTION


def baseline_mission():
    return R.Mission.implied_baseline()


def mission_form():
    """Field descriptors plus the implied baseline as defaults."""
    base = baseline_mission()
    fields = []
    for name, kind, unit, help_ in MISSION_FIELDS:
        fields.append({"name": name, "kind": kind, "unit": unit, "help": help_,
                       "default": getattr(base, name),
                       "choices": sorted(R.STROKE_BY_FIELD_CLASS)
                       if name == "field_class" else []})
    return {"fields": fields, "axes": list(R.priority_axes()),
            "materials": [R.PLA_FFF.name],
            "t_max_service_c": R.PLA_FFF.t_max_service_c}


def calibrated_points():
    """What portfolio `DEFAULT_WEIGHTS` is already running, in points.

    The panel opens on this rather than on an even 20/20/20/20/20, because the honest
    starting position is the one the tree is actually optimising -- and seeing it is how
    you find out that `phase_ripple` is running at zero.
    """
    smooth = MB.reference_smoothness(_reference_genome())
    pri, costs = R.calibrated_priorities(smooth)      # allocation and the c_T behind it
    return {"points": pri.as_dict(), "costs": costs, "smoothness_loss": smooth,
            "reference": os.path.basename(_reference_genome())}


def _mission_from(fields):
    base = baseline_mission()
    out = {}
    for name, kind, _u, _h in MISSION_FIELDS:
        if name not in fields or fields[name] in (None, ""):
            continue
        out[name] = int(fields[name]) if kind == "int" else (
            str(fields[name]) if kind == "choice" else float(fields[name]))
    return replace(base, **out)


def derive(fields, points=None):
    """Mission + points -> a requirement set, with every derivation shown.

    No solve. Milliseconds. This is `wheel_mbse` steps 1-4 and deliberately not step 5:
    MBSE_PLAN's own reason is that *a wrong requirement and a bad descent look the same
    from the outside*, so deriving must be cheap enough to do while typing.
    """
    mission = _mission_from(fields or {})
    priorities, smooth = None, None
    if points:
        smooth = MB.reference_smoothness(_reference_genome())
        priorities = R.Priorities({k: float(v) for k, v in points.items()})
    req = R.Requirements.from_mission(mission, priorities, smooth)

    default_w = dict(WO.DEFAULT_WEIGHTS)
    weights = []
    for key in sorted(set(default_w) | set(req.weights)):
        was, now = default_w.get(key), req.weights.get(key)
        weights.append({"term": key, "default": was, "value": now,
                        "moved": (was is None or now is None
                                  or abs(now - was) > 1e-12 * max(1.0, abs(was)))})

    return {
        "mission": mission.as_dict(),
        "derivations": [{"name": n, "unit": u, "why": why,
                         "value": float(getattr(mission, n))}
                        for n, u, why in DERIVATIONS],
        "requirements": {k: req.as_dict()[k] for k in
                         ("force_n", "target_deflection_mm", "allowable_stress_mpa",
                          "min_wall_mm", "e_mpa", "nu")},
        "req_hash": req.req_hash(),
        "weights": weights,
        "provenance": req.as_dict().get("provenance", {}),
        "points": priorities.as_dict() if priorities else None,
        "thermal": {"e_retention": R.PLA_FFF.e_retention(mission.ambient_c),
                    "sigma_retention": R.PLA_FFF.sigma_retention(mission.ambient_c),
                    "note": R.THERMAL_SCOPE_NOTE},
    }


def save_requirements(fields, points, path):
    """Write a requirements.json a Stage-3 run can be pointed at with --requirements."""
    mission = _mission_from(fields or {})
    priorities, smooth = None, None
    if points:
        smooth = MB.reference_smoothness(_reference_genome())
        priorities = R.Priorities({k: float(v) for k, v in points.items()})
    req = R.Requirements.from_mission(mission, priorities, smooth)
    req.save(path)
    return {"path": path, "req_hash": req.req_hash()}


def compliance(record_path=None, fields=None, points=None):
    """The compliance table for a genome record against a requirement set.

    Reads the record's ALREADY-SCORED numbers -- it does not solve. That is what makes it
    instant, and it is also its limit: the table describes the mesh and kinematics the
    record was produced under, not the ones the panel currently shows.
    """
    record_path = record_path or _reference_genome()
    with open(record_path) as fh:
        record = json.load(fh)
    if fields is None and points is None:
        req = R.Requirements.baseline()
        derived = None
    else:
        derived = derive(fields, points)
        mission = _mission_from(fields or {})
        priorities, smooth = None, None
        if points:
            smooth = MB.reference_smoothness(_reference_genome())
            priorities = R.Priorities({k: float(v) for k, v in points.items()})
        req = R.Requirements.from_mission(mission, priorities, smooth)

    if "loss_terms" not in record:
        return {"error": f"{os.path.basename(record_path)} carries no loss_terms block, "
                         f"so there is nothing to verify against.",
                "rows": [], "compliant": None}

    strict_refused = None
    try:
        table = R.verify(record, req, strict=True)
    except ValueError as exc:
        strict_refused = str(exc)
        table = R.verify(record, req, strict=False)

    table["record"] = os.path.basename(record_path)
    table["genome_hash"] = record.get("_hash") or wg.genome_hash(record["genes"])
    table["strict_refused"] = strict_refused
    table["derived"] = derived
    return table


def status():
    """What the tree looks like right now: genome, export freshness, gate residue."""
    out = {"root": ROOT}
    try:
        with open(_reference_genome()) as fh:
            rec = json.load(fh)
        out["genome"] = {
            "path": os.path.basename(_reference_genome()),
            "hash": rec.get("_hash") or wg.genome_hash(rec["genes"]),
            "mtime": os.path.getmtime(_reference_genome()),
            "genes": rec["genes"],
            "loss": rec.get("loss"),
            "metrics": rec.get("metrics", {}),
            "req_hash": (rec.get("search") or {}).get("req_hash"),
            "has_requirements_block": "requirements" in rec,
        }
    except (OSError, ValueError, KeyError) as exc:
        out["genome"] = {"error": str(exc)}

    manifest = os.path.join(PP.EXPORT, "wheel_step_manifest.json")
    try:
        with open(manifest) as fh:
            man = json.load(fh)
        stale = man.get("genome_hash") != out.get("genome", {}).get("hash")
        out["export"] = {
            "genome_hash": man.get("genome_hash"), "stale": stale,
            "exported_at": man.get("exported_at"),
            "export_seconds": man.get("export_seconds"),
            "mass_g": (man.get("solid") or {}).get("mass_g_pla"),
            "valid": (man.get("solid") or {}).get("valid"),
            # OCC does not always build the fillet that was asked for, and the manifest
            # is the only place that difference is recorded.
            "fillets": [{"requested": d.get("r_requested_mm"),
                         "built": d.get("r_built_mm"),
                         "kt_error_pct": d.get("kt_error_pct")}
                        for d in (man.get("fillets") or {}).get("detail", [])],
        }
    except (OSError, ValueError):
        out["export"] = None
    return out
