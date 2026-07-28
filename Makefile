# Two interpreters, on purpose.  env-opt runs the optimizer and (later) the FEA;
# env-cad runs the CadQuery STEP exporter.  They cannot be merged — see
# requirements-cad.txt.

PY_OPT := .venv-opt/bin/python
PY_CAD := .venv-cad/bin/python

.PHONY: help env env-opt env-cad test smoke ga elites stage3 m8bi5 m8bi6 export studies clean-pyc

help:
	@echo "make env      build both virtualenvs"
	@echo "make test     run the test suite in env-opt"
	@echo "make smoke    fast GA run (seconds) — proves the pipeline is wired up"
	@echo "make ga       full GA run (minutes), then hands off to the exporter"
	@echo "make export   rebuild wheel.step from the existing best_solution.json"
	@echo "make studies  the verification gates: spoke-mesh validity (M2a),"
	@echo "              full-wheel mesh (M2b), beam agreement (M3), full-wheel"
	@echo "              FEA (M4), geometric nonlinearity (M5), real contact"
	@echo "              (M6), gradients (M7), the Stage-3 objective (M8a),"
	@echo "              the Stage-3 optimizer (M8b-i)."
	@echo "              Each writes a JSON report and exits nonzero on failure."
	@echo "make elites   full GA run that also writes stage2_elites.json, the"
	@echo "              multi-start set Stage 3 begins from"
	@echo "make stage3   Stage-3 descent from best_solution.json, writing"
	@echo "              stage3_run.json as it goes and stage3_best.json at the end"
	@echo "make m8bi5    the two sections that QUALIFY M8b-i's infeasibility verdict:"
	@echo "              the stress QoI up the mesh ladder, and the same feasibility"
	@echo "              question asked from all 16 Stage-2 elites (~2 h at coarse)"
	@echo "make m8bi6    the stress p-norm up the same ladder at p = 2,4,8,16,30, to find"
	@echo "              which exponent (if any) gives the constraint a mesh-independent"
	@echo "              value.  ~14 min: the sweep costs no extra solve"

env: env-opt env-cad

env-opt:
	python3 -m venv .venv-opt
	$(PY_OPT) -m pip install --upgrade pip
	$(PY_OPT) -m pip install -r requirements-opt.txt

env-cad:
	python3 -m venv .venv-cad
	$(PY_CAD) -m pip install --upgrade pip
	$(PY_CAD) -m pip install -r requirements-cad.txt

test:
	$(PY_OPT) -m pytest

# --smoke keeps population and generations tiny.  The point is to exercise every code
# path end to end in seconds, not to produce a usable genome.
smoke:
	$(PY_OPT) wheel_fea.py --smoke

ga:
	$(PY_OPT) wheel_fea.py

# The same run, plus the final population's distinct genomes.  Stage 3 multi-starts from
# these; nothing else on disk records more than the single winner.
elites:
	$(PY_OPT) wheel_fea.py --dump-population

# Stage 3 proper: projected Adam on the FEA objective.  Serial, so the cost is
# roughly (steps x phases x 0.7 s) at `coarse` — see study_stage3.py's S10.
stage3:
	$(PY_OPT) wheel_stage3.py

# M8b-i.5.  Deliberately NOT in `studies`: these two sections are ~2 h at `coarse` on top
# of the gate's ~2 h 45 m, and they measure the WHEEL rather than the code — the answer
# does not change per commit, and a gate nobody can afford to run stops being run.
m8bi5:
	$(PY_OPT) study_stage3.py --sections mesh_convergence,multistart \
	    --out study_stage3_m8bi5.json

# M8b-i.6 step 1.  The mesh ladder again, at five Gauss-point p-norm exponents instead of
# one.  ~14 min, not 5x14: every exponent is read off the displacement field the adjoint
# has already converged, so the sweep adds no mesh, no Newton and no adjoint.  `multistart`
# is deliberately NOT here — S12 re-measures the wheel, and the wheel has not moved; what
# is open is whether the QUANTITY it was measured in has a value.
m8bi6:
	$(PY_OPT) study_stage3.py --sections mesh_convergence \
	    --ladder-p 2,4,8,16,30 --out study_stage3_pnorm.json

export:
	$(PY_CAD) wheel_step_export.py

# The milestone gates.  These are not tests — they produce measured reports whose
# numbers are quoted in CLAUDE.md — but they do exit nonzero when a gate fails, so
# they are safe to run in CI.
studies:
	$(PY_OPT) study_mesh_quality.py --samples 2000
	$(PY_OPT) study_wheel_mesh.py --samples 200
	$(PY_OPT) study_beam_agreement.py
	$(PY_OPT) study_wheel_fea.py
	$(PY_OPT) study_gnl.py
	$(PY_OPT) study_contact.py
	$(PY_OPT) study_gradient.py
	$(PY_OPT) study_objective.py
	$(PY_OPT) study_stage3.py

clean-pyc:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
