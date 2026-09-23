"""One 2D contact solve of the shipped genome, filleted: phase, plane, kinematics, config.
Saves the field for `band2d.py` and prints one JSON line (drop, patch centre and half-angle).

usage (repo root, .venv-opt, wheel_pool.PINNED_ENV):
    PYTHONPATH=src .venv-opt/bin/python studies/probe_3d/solve2d.py PHASE PLANE KIN OUT.npz [CFG]
"""
import sys, time, json, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "studies")
import wheel_wheel as WW, wheel_fem as F
from study_deflection_gci import load_genes
ph, plane, kin, out = float(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
cfg = sys.argv[5] if len(sys.argv) > 5 else "medium"
t0 = time.time()
g = load_genes("best_solution.json")
m = WW.build_wheel(g, WW.get_config(cfg), fillet=True, phase_deg=ph)
r = F.solve_wheel_contact(m, plane=plane, kinematics=kin)
rep = dict(cfg=cfg, phase=ph, plane=plane, kin=kin, n_el=m.n_elements, n_nodes=m.n_nodes,
           drop=float(r["axle_drop_mm"]), centre_deg=float(r["patch_centre_deg"]),
           half_deg=float(r["patch_half_deg"]), force=float(r["contact_force_n"]),
           secant_it=r["secant"]["iterations"], wall=time.time() - t0)
print(json.dumps(rep), flush=True)
np.savez(out, u=np.asarray(r["u"]), xy=np.asarray(m.coords), conn=np.asarray(m.conn),
         rep=np.array([json.dumps(rep)]))
