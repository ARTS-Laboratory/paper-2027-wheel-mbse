"""Dump the shipped genome's filleted 2D meshes (phase 0) for the 3D probe.

usage (from the repo root, env-opt): PYTHONPATH=src .venv-opt/bin/python studies/probe_3d/dump2d.py OUTDIR
"""
import sys, json, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "studies")
import wheel_wheel as WW, wheel_fem as F
from study_deflection_gci import load_genes
genes = load_genes("best_solution.json")
for cfg in ("coarse", "medium"):
    m = WW.build_wheel(genes, WW.get_config(cfg), fillet=True, phase_deg=0.0)
    xy = np.asarray(m.coords); tie = np.asarray(m.node_sets["hub_tie"]); ro = np.unique(m.edge_sets["rim_outer"])
    r = np.hypot(*xy.T)
    area = F.gauss_volume(xy, m.conn, order=m.cfg.order, width=1.0)
    print(cfg, "el", m.n_elements, "nodes", m.n_nodes, "area %.4f" % np.sum(area), "x22.4 = %.2f" % (22.4*np.sum(area)),
          "tie r [%.5f, %.5f]" % (r[tie].min(), r[tie].max()), "rim_outer r [%.5f,%.5f]" % (r[ro].min(), r[ro].max()),
          "r range [%.4f, %.4f]" % (r.min(), r.max()))
    np.savez(f"{sys.argv[1]}/mesh2d_{cfg}.npz", xy=xy, conn=np.asarray(m.conn), tie=tie, rim_outer=ro, area=np.asarray(area))
