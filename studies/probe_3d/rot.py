"""Rotate a STEP about the z axis by +PHASE degrees (counter-clockwise), which is exactly what
`build_wheel(phase_deg=PHASE)` does to the phase-0 2D mesh (checked: 4.8e-14 mm).
usage: rot.py IN.step PHASE OUT.step"""
import sys, math, gmsh
gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
v = gmsh.model.occ.importShapes(sys.argv[1])
gmsh.model.occ.rotate(v, 0, 0, 0, 0, 0, 1, math.radians(float(sys.argv[2])))
gmsh.model.occ.synchronize(); gmsh.write(sys.argv[3]); gmsh.finalize()
