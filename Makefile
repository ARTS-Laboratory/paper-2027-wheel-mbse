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

# `minwall-%` is deliberately NOT here: GNU make excludes phony targets from implicit and
# pattern rule search, so listing the four arms would silently disable the rule that builds
# them ("Nothing to be done for 'minwall-1.6'").  Nothing on disk is named `minwall-1.6` —
# the arms write `stage3_minwall_<floor>.json` — so the rule fires without it.
.PHONY: help env env-opt env-cad test smoke ga elites stage3 m8bi5 m8bi6 m8bii1 m9 m9buck hubcap prod9 prod10 export svk svk-shipped svk-elite10 svk-medium buildcap studies clean-pyc

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
	@echo "make hubcap   the analytic hub-fillet cap against what OCC accepts: the"
	@echo "              inter-spoke void by ring classification, and the fillet"
	@echo "              acceptance threshold by bisection.  Needs BOTH envs, ~8 min"
	@echo "make prod9    PLAN.md 0(b): the production descent from elite 9.  ~4 h"
	@echo "make prod10   the same from elite 10.  RUN THE TWO SEQUENTIALLY, and on a"
	@echo "              memory-bound box launch each under a systemd-run cap: one"
	@echo "              descent holds ~12.7 GB anon and two do not fit in 31 GB."
	@echo "              Override PROD_STEPS/PROD_WORKERS/PROD_FIDELITY"
	@echo "make svk      SVK_PLAN.md step 3: re-scores the shipped genome and its"
	@echo "              rivals under BOTH kinematics with no optimizer, and reports"
	@echo "              the utilisation the CONSTRAINT computes rather than §14's"
	@echo "              p99-scaled estimate. Reproduces §14's load ladder first as"
	@echo "              a control. Override SVK_CONFIG/SVK_WORKERS"
	@echo "make svk-shipped | svk-elite10"
	@echo "              SVK_PLAN.md step 5: the 300-step descent under SVK, from"
	@echo "              each converged genome. ~5.3 h each. RUN THEM SEQUENTIALLY"
	@echo "              and capped, exactly as prod9/prod10 — see the comment there."
	@echo "              Override SVK_DESCENT_STEPS/SVK_DESCENT_WORKERS/SVK_MIN_WALL"
	@echo "make minwall-1.6 | -1.8 | -2.0 | -2.2"
	@echo "              PLAN.md 0(2): what the printable wall floor costs in grams."
	@echo "              125 steps from the elite-10 answer at each floor; 2.0 is the"
	@echo "              CONTROL arm and must be run for the others to be readable."
	@echo "              Override MINWALL_STEPS/MINWALL_WORKERS/MINWALL_GENOME"

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

# M9 Phase 2.  Measurement-only: tangent eigenvalue refinement, phase and load ladders.
m9:
	$(PY_OPT) -u studies/study_m9.py --out study_m9.json

# PLAN.md §0(a).  The analytic hub-fillet cap against what OCC actually accepts: the void
# measured on the profile by ring classification, and the fillet acceptance threshold found
# by BISECTION rather than read off the ladder (the ladder's rungs straddle the cap, so
# "the largest rung below it" is measurably the wrong criterion — see the driver).
#
# Out of `studies` for the m8bi5 reason plus one more: it needs BOTH interpreters, and what
# it measures is OCC's behaviour on this shape rather than anything a commit changed.
hubcap:
	$(PY_OPT) studies/study_hub_cap.py --out study_hub_cap.json

# M9 Phase 3.  The GENERALISED buckling load factor -- det(K_0 + lambda*K_g) = 0 under SVK
# kinematics -- which converges under refinement where `make m9`'s lambda_min(K_t) does not
# (that quantity has no K_t in it at all: wheel_contact_problem defaults to
# kinematics="linear", so the displacement threaded into assemble_stiffness is ignored).
#
# THE STATE MUST BE SOLVED UNDER SVK AS WELL AS THE OPERATOR ASSEMBLED UNDER IT.  Measured:
# an SVK stiffness assembled at a LINEAR-converged displacement reads 1.800 / 1.785 at
# smoke / coarse against the correct 1.378 / 1.360 -- a +31% error of exactly the kind this
# milestone exists to remove.  See the driver's header.
#
# Out of `studies` for the m8bi5 reason: it measures the WHEEL, not the commit, and it is
# ~40 min.  Measurement-only -- nothing here reaches the Stage-3 objective, `buckling`
# stays inert, and no threshold is invented.
m9buck:
	$(PY_OPT) -u studies/study_m9_buckling.py --out study_m9_buckling.json

# PLAN.md §0(b).  The production multi-start descent, from elites 9 and 10 rather than
# best_solution.json — that genome is a GA optimum for the BEAM surrogate and sits at
# -25.43% deflection error, while 9 and 10 are the two designs already inside the feasible
# box on both constraints.  The objective it descends is MASS: both constraints are
# satisfiable together and every barrier is flat at a feasible design, so mass is the only
# term with anything left to give.  No flag selects that — it is what the existing weights
# already do once deflection is met.
#
# ONE TARGET PER START, AND THEY RUN SEQUENTIALLY.  This comment used to argue the
# opposite — S13 measures 4 workers at 2.93x / 0.73 efficiency against 8 at 3.95x / 0.49,
# so two quarter-box runs look like they beat two half-box runs on the same 8 cores.  That
# is CPU arithmetic on a box that runs out of MEMORY first, and it was tried: one descent
# at `coarse` with 4 workers sits flat at ~12.7 GB anon (parent ~4.5 GB, four workers
# ~2 GB each), so two of them is ~25 GB against 31 GB of RAM and a 2 GB swapfile.  What
# that costs is not a slow run, it is the DESKTOP — `systemd-oomd` kills the whole
# `user@1000.service` slice at 50% pressure for 20 s, dropping the user to the login
# screen and taking every terminal with it.  Launch each capped and detached instead:
#
#   systemd-run --user --unit=wheel-prod9 -p MemoryMax=20G --collect \
#       --working-directory=$$PWD /usr/bin/make prod9
#
# ~4.0 h of descent each (300 steps x 47.6 s), plus the fidelity checks below.
#
# Out of `studies` for the m8bi5 reason: this is a SEARCH, not a gate.  Nothing here has a
# pass/fail verdict, and it is hours.
#
# DRIVEN BY MAKE RATHER THAN BY HAND, and that is not a convenience.  The five pinned vars
# at the top of this file are a CORRECTNESS setting for the PARENT as much as for the
# workers — the parent runs T1, T2, the Kt cache and the reduction over phase replies — and
# `wheel_stage3.py` does not set them itself.  A worker pins itself before its first import
# (`wheel_pool_worker.py`); a parent launched straight from a shell does not, which both
# costs the adjoint its bit-reproducibility and oversubscribes the box with a 16-thread XLA
# pool alongside 8 pinned workers.  See `wheel_pool.PINNED_ENV`.
#
# The distinct --out AND --best-out names are load-bearing for the same concurrency: both
# default to a fixed name under the repo root and `main()` writes --best-out
# unconditionally, so two simultaneous runs at the defaults would silently clobber each
# other's genome.
# `-u` on the interpreter, and it is what makes a DETACHED run observable at all.  Python
# block-buffers stdout when it is not a TTY, and under `systemd-run` it is a journal socket,
# so a 4 h descent emits NOTHING to `journalctl` until the buffer fills or the process
# exits — measured: 162 steps in, zero `[step ...]` lines in the unit's journal.  That makes
# `journalctl --user -u wheel-prod10 -f` useless for progress and, worse, hides a traceback
# until exit.  Unbuffered costs nothing here (one print per ~47 s step) and moves no number.
PROD_STEPS ?= 300
PROD_WORKERS ?= 4

# `uniform`, NOT the `rqmc` default, and this is a MEMORY constraint rather than a
# statistical preference.  `rqmc` redraws the stencil every step from an `n_sub`-point
# sub-lattice, so a 300-step run visits `n_phase * n_sub` = 64 distinct phase values —
# and `wheel_wheel.coord_fn` keys its jit cache on `float(phase)` with FIFO eviction only
# at 128 entries, so all 64 traces are RETAINED.  Measured: ~0.4 GB per trace, and the
# pool holds all 64 no matter how the slots are split across workers, so `--workers` is
# not a lever on it.  A run at the defaults peaked at 18.9 GB and was OOM-killed at step 3.
# `uniform` fixes the 8 phases, so the cache saturates at 8 traces after step 0.
PROD_SCHEME ?= uniform

# OFF, and that default is a MEASUREMENT rather than a preference.  The check is a PURE
# OBSERVATION at `medium` and it runs on a wholly separate SERIAL evaluator, never the
# phase pool (deliberately — `wheel_stage3.descend`), so it pays all 8 phases end to end
# with no parallelism.  Measured from the first production attempt's
# `settings.fidelity_check_solve_s`: ONE check is 604.6 s, so every-50 over 300 steps is
# 7 checks ~= 1.2 h, not the ~45 min this comment used to claim.  It also costs ~3.4 GB
# PERMANENTLY — that second `Evaluator` retains its own jit cache and mesh for the life of
# the run, taking the parent 4.5 -> 7.9 GB across the FIRST check and leaving it there.
# On a memory-bound box that is the first thing to turn off, and it is what both recorded
# production runs did.  Dropping it costs evidence, never trajectory: the gradient it
# returns is discarded and can never reach m/v/delta/z.  Set PROD_FIDELITY=50 to restore.
PROD_FIDELITY ?= 0

prod9:
	$(PY_OPT) -u src/wheel_stage3.py --start rank:9 --steps $(PROD_STEPS) \
	    --workers $(PROD_WORKERS) --phase-scheme $(PROD_SCHEME) \
	    --fidelity-check-every $(PROD_FIDELITY) --fidelity-check-config medium \
	    --out stage3_prod_elite9.json --best-out stage3_prod_best_elite9.json

prod10:
	$(PY_OPT) -u src/wheel_stage3.py --start rank:10 --steps $(PROD_STEPS) \
	    --workers $(PROD_WORKERS) --phase-scheme $(PROD_SCHEME) \
	    --fidelity-check-every $(PROD_FIDELITY) --fidelity-check-config medium \
	    --out stage3_prod_elite10.json --best-out stage3_prod_best_elite10.json

# PLAN.md 0(2).  What the printable wall floor COSTS, in grams.
#
# The production descents drove all four thickness genes onto MIN_WALL_MM and left them
# there, so a manufacturing constant -- not the FEA, not the deflection target, not the
# stress constraint -- sets 4 of the 14 genes at the answer, and every gram below 58.660
# is on the far side of it.  That constant sat on "the decision that is a human's" with no
# quantification at all, which is not a decidable state.  Four floors turn it into a slope.
#
# THE START IS THE ELITE-10 ANSWER, NOT `rank:10`.  Starting from the converged genome is
# what makes 125 steps enough: measured off elite 10's own record, step 100 is +0.190% from
# its final loss, step 125 +0.096%, step 150 +0.048%.  A tenth of a percent is far finer
# than 0.2 mm of floor will move the mass, so the sweep resolves the thing it is asking
# about at a third of the cost of a full descent.
#
# 2.0 IS THE CONTROL ARM AND IT IS NOT PADDING.  It restarts from its own converged answer
# at its own floor, so it should barely move; whatever it DOES move is the cost of the
# fresh Adam state (m/v reset) plus 125 steps.  Without it the sweep cannot separate
# "0.2 mm of floor" from "125 more steps", and the other three arms are unreadable.
#
# `uniform` for the reason PROD_SCHEME gives, plus one that outlives the memory argument:
# the start point and the control arm were both measured under it, and changing the
# quadrature at the same time as the floor would measure neither.
MINWALL_STEPS ?= 125
MINWALL_WORKERS ?= 4
MINWALL_GENOME ?= stage3_prod_best_elite10.json

minwall-%:
	$(PY_OPT) -u src/wheel_stage3.py --start best --genome $(MINWALL_GENOME) \
	    --min-wall $* --steps $(MINWALL_STEPS) --workers $(MINWALL_WORKERS) \
	    --phase-scheme uniform --fidelity-check-every 0 \
	    --out stage3_minwall_$*.json --best-out stage3_minwall_best_$*.json

# EXPORT_GENOME builds a genome that is NOT the shipped one — a promotion candidate,
# say — and the exporter then names its artifacts after that file's stem rather than
# `wheel.*`, so `make export EXPORT_GENOME=stage3_minwall_best_1.2.json` cannot
# overwrite the shipped STEP.  Empty (the default) is the shipped genome, unchanged.
EXPORT_GENOME ?=

export:
	$(PY_CAD) src/wheel_step_export.py $(if $(EXPORT_GENOME),--genome $(EXPORT_GENOME))

# SVK_PLAN.md step 3.  Re-scores the shipped genome and its rivals under BOTH kinematics
# with no optimizer, and answers the one question §14 left as an estimate: is the shipped
# wheel still feasible under SVK, at the utilisation the CONSTRAINT computes rather than
# the p99-scaled figure §14 quoted.
#
# DELIBERATELY NOT IN `studies`, for the reason `m8bi5`, `m9buck` and `hubcap` are not:
# it measures THE WHEEL, NOT THE COMMIT.  Its answer does not move when the code changes,
# it is the better part of an hour at `medium`, and a gate nobody can afford to run stops
# being run.  SVK_WORKERS is the memory cap and nothing else sizes it — see PLAN.md §1.
SVK_CONFIG ?= medium
SVK_WORKERS ?= 4
# Step 6 re-scores the descent winner at `medium` before promoting it, because Step 5
# descended at `coarse` and the two rungs differ by ~1.1% on this wheel.  SVK_EXTRA is
# additive (`label=path,...`) and SVK_ONLY narrows the built-in set, so the winner can be
# scored WITHOUT editing GENOMES — the bare `make svk` must keep reproducing Step 3.
SVK_EXTRA ?=
SVK_ONLY ?=
SVK_OUT ?= study_svk_rescore.json

svk:
	$(PY_OPT) -u studies/study_svk_rescore.py --config $(SVK_CONFIG) \
	    --workers $(SVK_WORKERS) --out $(SVK_OUT) \
	    $(if $(SVK_ONLY),--only $(SVK_ONLY),) $(if $(SVK_EXTRA),--extra $(SVK_EXTRA),)

# SVK_PLAN.md step 5.  The descent itself, under St Venant-Kirchhoff.
#
# TWO STARTS, RUN SEQUENTIALLY, for the reason the `prod9`/`prod10` block above argues at
# length and which SVK does not soften: the box runs out of MEMORY before it runs out of
# cores.  Step 2 measured a 4-worker SVK descent at 13.16 GiB peak anon against linear's
# 12.56 — only 1.05x, but two at once is still ~26 GB against 31, which is how the desktop
# got taken down twice.  Launch each capped and detached:
#
#   systemd-run --user --unit=wheel-svk-shipped -p MemoryMax=16G --collect \
#       --working-directory=$$PWD /usr/bin/make svk-shipped
#
# 16G, not Step 2's measured 13.16: the cap is a KILL SWITCH, not a budget, and it wants
# enough headroom that a normal run never touches it.
#
# ~5.3 h each (300 steps x 62.3 s), against ~3.9 h for the same descent under linear.
# That 1.36x is Step 2's measurement and it is NOT extra Newton iterations — those go
# 26.00 -> 26.75, and backtracks actually FALL.  It is the SVK tangent assembly and the
# nonlinear vJP, so no solver tolerance buys it back.
#
# BOTH STARTS ARE `--start best --genome <file>`, NOT `rank:N`.  The elite ranks are
# Stage-2 answers scored under a beam surrogate; what this arc needs to know is where the
# two best CONVERGED genomes go when the strain measure under them changes, and starting
# from an unconverged point would confound "SVK moved it" with "300 more steps moved it".
#
# --min-wall 1.2 because that is the floor the shipped genome was promoted at (PLAN.md
# §14) and the sweep that chose it (`minwall-`) ran under linear.  Re-deriving the floor
# and changing the kinematics in the same run would measure neither.
#
# `uniform` and `--fidelity-check-every 0` for the reasons PROD_SCHEME and PROD_FIDELITY
# give above; both apply here unchanged, and the fidelity check's second serial Evaluator
# would cost 1.36x more under SVK than the 604.6 s already measured.
#
# Distinct --out AND --best-out per start, load-bearing for the same reason prod9/prod10's
# are: both default to a fixed name under the repo root.
SVK_DESCENT_STEPS ?= 300
SVK_DESCENT_WORKERS ?= 4
SVK_MIN_WALL ?= 1.2

svk-shipped:
	$(PY_OPT) -u src/wheel_stage3.py --start best --genome best_solution.json \
	    --kinematics svk --min-wall $(SVK_MIN_WALL) \
	    --steps $(SVK_DESCENT_STEPS) --workers $(SVK_DESCENT_WORKERS) \
	    --phase-scheme uniform --fidelity-check-every 0 \
	    --out stage3_svk_shipped.json --best-out stage3_svk_best_shipped.json

svk-elite10:
	$(PY_OPT) -u src/wheel_stage3.py --start best \
	    --genome stage3_prod_best_elite10.json \
	    --kinematics svk --min-wall $(SVK_MIN_WALL) \
	    --steps $(SVK_DESCENT_STEPS) --workers $(SVK_DESCENT_WORKERS) \
	    --phase-scheme uniform --fidelity-check-every 0 \
	    --out stage3_svk_elite10.json --best-out stage3_svk_best_elite10.json

# SVK_PLAN.md step 6.  RE-CONVERGE THE WINNER AT `medium`.
#
# Both step-5 descents ran at `coarse` and converged their deflection to -0.04%.  At
# `medium` the same genome reads +1.65%, against a +/-0.3% gate — so the coarse answer does
# not promote, and the fix is to meet the existing gate at the finer rung rather than to
# move the gate.  See the STEP 6 block in SVK_PLAN.md.
#
# 100 steps, not 300: this warm-starts 1.65% from target, not 19.7%.  One value+grad at
# medium/4-workers measured 273 s against coarse's 58.6 s (4.7x), so this is ~7.6 h.
#
# FIDELITY CHECK ON, pointing back at `coarse`: step 5 ran it off and that is exactly why
# the rung gap was found after 9.8 h of descending instead of at step 0.  It is a pure
# observation and cannot redirect the descent — it only puts both rungs on the record.
SVK_MEDIUM_STEPS ?= 100

svk-medium:
	$(PY_OPT) -u src/wheel_stage3.py --start best \
	    --genome stage3_svk_best_shipped.json \
	    --config medium --kinematics svk --min-wall $(SVK_MIN_WALL) \
	    --steps $(SVK_MEDIUM_STEPS) --workers $(SVK_DESCENT_WORKERS) \
	    --phase-scheme uniform \
	    --fidelity-check-every 25 --fidelity-check-config coarse \
	    --out stage3_svk_medium.json --best-out stage3_svk_best_medium.json

# BUILD_PLAN.md step 5.  RE-DESCEND WITH A CAP THAT CAN SEE THE HUB ARRIVAL ANGLE.
#
# Same rung, same kinematics, same floor and same scheme as `svk-medium` above — the ONLY
# thing that differs is `wheel_objective.hub_fillet_cap_mm`, which now takes the arrival.
# That is the point: `svk-medium`'s winner is the control, this is the treatment, and any
# other difference between them would make the pair unreadable.
#
# WARM-STARTS FROM `svk-medium`'S OWN WINNER, `bc77614`, and starting from the UNBUILDABLE
# design is deliberate rather than convenient.  It is 0.04% from the deflection target and
# 12/24 on the exporter's hub corners, so the question this run asks is the sharp form of
# the arc's question: given a cap that can finally see why, does the descent walk back off
# the geometry it walked onto?  Starting from the shipped genome, which violates the new
# cap by 1.2%, would ask a much easier one.
#
# DISTINCT --out AND --best-out, load-bearing for the reason prod9/prod10's are, and doubly
# so here: clobbering `stage3_svk_best_medium.json` would destroy the control.
#
# ~226 s/step measured on the `svk-medium` run this mirrors (22627.5 s / 100), so ~6.3 h.
BUILDCAP_STEPS ?= 100
# Variables, not literals, ONLY so step 5 can be re-run against a moved gene box without
# editing the recipe — the warm start stays the control either way.  BUILD_PLAN.md step 5b.
BUILDCAP_GENOME ?= stage3_svk_best_medium.json
BUILDCAP_OUT ?= stage3_buildcap_medium.json
BUILDCAP_BEST ?= stage3_buildcap_best_medium.json

buildcap:
	$(PY_OPT) -u src/wheel_stage3.py --start best \
	    --genome $(BUILDCAP_GENOME) \
	    --config medium --kinematics svk --min-wall $(SVK_MIN_WALL) \
	    --steps $(BUILDCAP_STEPS) --workers $(SVK_DESCENT_WORKERS) \
	    --phase-scheme uniform \
	    --fidelity-check-every 25 --fidelity-check-config coarse \
	    --out $(BUILDCAP_OUT) --best-out $(BUILDCAP_BEST)

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
