# Two interpreters, on purpose.  env-opt runs the optimizer and (later) the FEA;
# env-cad runs the CadQuery STEP exporter.  They cannot be merged — see
# requirements-cad.txt.

PY_OPT := .venv-opt/bin/python
PY_CAD := .venv-cad/bin/python

# The modules live in src/ and are imported flat (`import wheel_fea as W`), so every
# interpreter this Makefile starts — and every subprocess THEY start — needs src/ on
# the path.  Exported rather than set per-recipe because the CAD hand-off in
# wheel_fea.py spawns .venv-cad itself.
export PYTHONPATH := $(CURDIR)/src

# The parallelism in this tree lives at the PHASE level (wheel_pool.py), not inside BLAS
# and not inside XLA.  N phase workers each spinning up a core-count-sized thread pool
# oversubscribes any machine by roughly N x.
#
# XLA_FLAGS IS NOT A PERFORMANCE KNOB, IT IS WHAT MAKES THE ADJOINT REPRODUCIBLE.
# Measured: two plain serial runs of one `coarse` adjoint in two separate interpreters,
# no pool involved, agree on every forward value to the bit and disagree on the GRADIENT
# by 3.33e-16 — XLA's CPU intra-op thread pool is sized from the machine and its parallel
# reductions do not associate the same way twice.  Pinned, the same comparison is exactly
# zero, and it costs nothing: 19.84 s against 20.43 s for one `coarse` phase.
# study_stage3.py's S13 gates pooled == serial EXACTLY, which is only a meaningful claim
# because of this line.  See `wheel_pool.PINNED_ENV`.
#
# `?=` rather than `:=`: someone who sets these deliberately is driving, and S13 will tell
# them what it cost.  `conftest.py` sets the same five so bare `pytest` matches `make test`.
OMP_NUM_THREADS ?= 1
MKL_NUM_THREADS ?= 1
OPENBLAS_NUM_THREADS ?= 1
NUMEXPR_NUM_THREADS ?= 1
XLA_FLAGS ?= --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1
export OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS XLA_FLAGS

.PHONY: help env env-opt env-cad test smoke ga elites stage3 m8bi5 m8bi6 m8bii1 export studies clean-pyc

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
	@echo "              (add --workers -1 to run the phase loop across processes)"
	@echo "make m8bi5    the two sections that QUALIFY M8b-i's infeasibility verdict:"
	@echo "              the stress QoI up the mesh ladder, and the same feasibility"
	@echo "              question asked from all 16 Stage-2 elites (~2 h at coarse)"
	@echo "make m8bi6    the stress p-norm up the same ladder at p = 1,2,3,4,6,8,12,16,24,30,"
	@echo "              to find which exponent (if any) gives the constraint a"
	@echo "              mesh-independent value.  ~14 min: the sweep costs no extra solve"
	@echo "make m8bii1   S13: one 8-phase evaluation serial and pooled, up a worker"
	@echo "              ladder sized to this machine.  Gates that the two answers are"
	@echo "              bit-identical, and reports what the parallelism buys"

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
	$(PY_OPT) src/wheel_fea.py --smoke

ga:
	$(PY_OPT) src/wheel_fea.py

# The same run, plus the final population's distinct genomes.  Stage 3 multi-starts from
# these; nothing else on disk records more than the single winner.
elites:
	$(PY_OPT) src/wheel_fea.py --dump-population

# Stage 3 proper: projected Adam on the FEA objective.  Serial, so the cost is
# roughly (steps x phases x 0.7 s) at `coarse` — see study_stage3.py's S10.
stage3:
	$(PY_OPT) src/wheel_stage3.py

# M8b-i.5.  Deliberately NOT in `studies`: these two sections are ~2 h at `coarse` on top
# of the gate's ~2 h 45 m, and they measure the WHEEL rather than the code — the answer
# does not change per commit, and a gate nobody can afford to run stops being run.
m8bi5:
	$(PY_OPT) studies/study_stage3.py --sections mesh_convergence,multistart \
	    --out study_stage3_m8bi5.json

# M8b-i.6 step 1.  The mesh ladder again, at ten Gauss-point p-norm exponents instead of
# one.  ~14 min, not 10x14: every exponent is read off the displacement field the adjoint
# has already converged, so the sweep adds no mesh, no Newton and no adjoint.  `multistart`
# is deliberately NOT here — S12 re-measures the wheel, and the wheel has not moved; what
# is open is whether the QUANTITY it was measured in has a value.
#
# THE TWO BOOKENDS ARE NOT PADDING.  `p=1` is the anchor: the norm is normalized by the
# volume (`wheel_adjoint._qoi_pnorm_stress`), so p=1 is the volume-weighted MEAN von Mises,
# a plain quadrature of an integral, and it MUST converge — if it does not, the exponent is
# not the problem and lowering `p` cannot fix the constraint.  `p=30` is the shipped default
# and must reproduce the constraint's own series exactly, which is what validates every
# other row.  The dense middle locates the knee rather than bracketing it.
m8bi6:
	$(PY_OPT) studies/study_stage3.py --sections mesh_convergence \
	    --ladder-p 1,2,3,4,6,8,12,16,24,30 --out study_stage3_pnorm.json

# M8b-ii item 1.  S13: the same `coarse` 8-phase evaluation serial and pooled, up a worker
# ladder derived from THIS machine.  ~15 min on 16 cores; longer on fewer, because the
# ladder is shorter but each rung is slower.
#
# Out of `studies` for the m8bi5 reason and one more: the wall-clock half of this report
# describes the machine it ran on, so a committed number is evidence about a host rather
# than about a commit.  The half that does travel — pooled == serial, EXACTLY — is in
# `make test` (tests/test_pool.py) where it belongs, and runs every time.
m8bii1:
	$(PY_OPT) studies/study_stage3.py --sections phase_pool \
	    --out study_stage3_pool.json

export:
	$(PY_CAD) src/wheel_step_export.py

# The milestone gates.  These are not tests — they produce measured reports whose
# numbers are quoted in CLAUDE.md — but they do exit nonzero when a gate fails, so
# they are safe to run in CI.
studies:
	$(PY_OPT) studies/study_mesh_quality.py --samples 2000
	$(PY_OPT) studies/study_wheel_mesh.py --samples 200
	$(PY_OPT) studies/study_beam_agreement.py
	$(PY_OPT) studies/study_wheel_fea.py
	$(PY_OPT) studies/study_gnl.py
	$(PY_OPT) studies/study_contact.py
	$(PY_OPT) studies/study_gradient.py
	$(PY_OPT) studies/study_objective.py
	$(PY_OPT) studies/study_stage3.py

clean-pyc:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
