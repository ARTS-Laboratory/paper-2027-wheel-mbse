"""`wheel_adjoint.rim_band_surface_stress` on fields saved by `solve2d.py`, one JSON line each.

usage (repo root, .venv-opt): .venv-opt/bin/python studies/probe_3d/band2d.py R.npz [...]
"""
import sys, json, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "studies")
import wheel_wheel as WW, wheel_fem as F, wheel_adjoint as WA
from study_deflection_gci import load_genes
g = load_genes("best_solution.json")
for f in sys.argv[1:]:
    d = np.load(f); rep = json.loads(str(d["rep"][0])); cfg = rep.get("cfg", "medium")
    m = WW.build_wheel(g, WW.get_config(cfg), fillet=True, phase_deg=rep["phase"])
    assert np.array_equal(np.asarray(m.coords), d["xy"])
    prob = F.wheel_contact_problem(m, indentation_mm=rep["drop"], kinematics=rep["kin"], plane=rep["plane"])
    v = WA.rim_band_surface_stress(prob, d["u"], m)
    print(json.dumps(dict(cfg=cfg, phase=rep["phase"], plane=rep["plane"], kin=rep["kin"], band_od=v)), flush=True)
