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
.PHONY: help env env-opt env-cad test smoke ga elites stage3 m8bi5 m8bi6 m8bii1 m9 m9buck hubcap prod9 prod10 export svk svk-shipped svk-elite10 svk-medium buildcap knee kinrank contact gci corner corner-fillet junction fillet filletblock filletcost filletterms filletoptimum filletkt filletpnorm filletpnormbox filletconda filletwiring triblock trirule tribend reds reds-ratio reds-hub reds-hub-fillet reds-hub-fillet-rungs boundarywaste stage3-resume mbse mbsebase mbsecal mbsescore studies clean-pyc

help:
	@echo "make env      build both virtualenvs"
	@echo "make test     run the test suite in env-opt"
	@echo "make smoke    fast GA run (seconds) — proves the pipeline is wired up"
	@echo "make ga       full GA run (minutes), then hands off to the exporter"
	@echo "make export   rebuild wheel.step from the existing best_solution.json"
	@echo "make mbse     MBSE_PLAN Step 7: a mission and a 100-point allocation in,"
	@echo "              a requirement set, a compliance table and (with --descend)"
	@echo "              a re-optimised genome out.  Seconds without --descend."
	@echo "              Override MBSE_ARGS, e.g. MBSE_ARGS='--ambient-c 40'"
	@echo "make mbsebase MBSE_PLAN Step 0: the derivations run BACKWARDS — what"
	@echo "              all-up weight, sink rate, ambient and service life the four"
	@echo "              shipped constants imply.  Solves nothing, ~1 s"
	@echo "make mbsecal  MBSE_PLAN Step 4: what portfolio DEFAULT_WEIGHTS is already"
	@echo "              running, in points, with the loss-share reading refuted and"
	@echo "              the promotion chain as a hold-out.  Solves nothing, ~1 s"
	@echo "make mbsescore MBSE_PLAN Step 5: the shipped wheel against five named"
	@echo "              requirement profiles, with a compliance table each.  Gates"
	@echo "              that req=baseline() is bit-identical to naming nothing AND"
	@echo "              that at least one profile comes back NON-COMPLIANT"
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
	@echo "make knee     DEFECT8_PLAN.md step 4: the production descent under the"
	@echo "              knee'd stress_margin. Every knob is §19's, so §19's own run"
	@echo "              (stage3_margin_medium.json) is an exact control and the"
	@echo "              objective is the only difference. ~6.3 h, capped as prod9"
	@echo "make kinrank  KINEMATICS_PLAN.md step 1: scores every distinct committed"
	@echo "              genome under BOTH kinematics and asks whether linear RANKS"
	@echo "              designs the way SVK does — argmin identity, Spearman rho and"
	@echo "              the gradient cosine, against bars registered before the run."
	@echo "              ~1 h at coarse. Override KINRANK_CONFIG/KINRANK_WORKERS"
	@echo "make contact  CONTACT_PLAN.md step 2: ONE cell of the patch-resolution"
	@echo "              matrix — is the axle drop the objective steers by still"
	@echo "              mesh-convergent on the genome that ships? Override"
	@echo "              CONTACT_GENOME/CONTACT_KIN/CONTACT_SECTIONS/CONTACT_OUT."
	@echo "              NOT the M6 gate: that is study_contact.py in full, in studies"
	@echo "make gci      SVK_PLAN's closing item: the ±0.3% deflection gate is"
	@echo "              satisfiable at exactly one rung. Runs the gate's OWN QoI"
	@echo "              (axle_drop_mean_mm, 8-phase, both kinematics) up the whole"
	@echo "              mesh ladder and extrapolates it. 1 h 35 m, 20.6 GB peak."
	@echo "              --reanalyse redoes the arithmetic on a saved report free"
	@echo "make corner   is the junction corner a real stress singularity? measures"
	@echo "              the field AT the corner — radial decay, and whether the"
	@echo "              peak diverges under refinement — against Williams' wedge"
	@echo "              eigenvalue. 8.5 s for the whole ladder (PLAN §30)"
	@echo "make corner-fillet  the SAME ladder on the FILLETED mesh — FILLET_PLAN.md"
	@echo "              Step 2. Adds the fillet's own probes (both tangent points,"
	@echo "              the arc, and the peak over the whole arc surface), the"
	@echo "              R -> 0 control that says it is the same wheel, and what"
	@echo "              each candidate LAYER PROFILE costs the deflection's"
	@echo "              convergence — the two-objective half of PART 13's"
	@echo "              declined call. ~110 s"
	@echo "make filletblock  can the fillet BE a block, and can the sector be"
	@echo "              blocked around it? the region PART 3 named has two cusps;"
	@echo "              the boundary-layer block that meshes; and the whole"
	@echo "              filleted sector, 11 blocks and 14 whole-edge seams, with"
	@echo "              what it costs the ring; plus the sector-fit MARGIN that"
	@echo "              predicts every refusal, what clamping to it buys, and"
	@echo "              the FOLD gate — the closed-form margin that catches the"
	@echo "              one genome whose spoke does not exist, and how often the"
	@echo "              mesh-based filter it replaces leaks. ~230 s, geometry only"
	@echo "make filletcost  what the FILLETED objective costs — §88's unmeasured"
	@echo "              \"2-3x\", measured. three altitudes (mesh build; one solve"
	@echo "              and one adjoint, G10's method on the mesh G10 never ran"
	@echo "              on; one 8-phase objective evaluation) at both kinematics,"
	@echo "              all post-trace with the trace priced separately. the SVK"
	@echo "              evaluation row is the one the decision reads. exits 0"
	@echo "make filletterms  WHERE the filleted objective's 17x loss difference"
	@echo "              lives — §90 recorded 671.66 against 38.79 at the shipped"
	@echo "              genome and did not attribute it. breakdown['terms'] on"
	@echo "              both meshes, same genome: T1+T2 (no solve, no kinematics)"
	@echo "              then the whole 8-phase evaluation at both kinematics."
	@echo "              tests §90's own \"if the barrier is most of it\". exits 0"
	@echo "make filletoptimum  what the switch does to the OPTIMUM, not to the"
	@echo "              score. §91 found the shipped genome sits 50.43% under"
	@echo "              its deflection target on the filleted mesh against"
	@echo "              4.945% on the unfilleted one, so the question is"
	@echo "              whether wiring the fillet in is a re-score or a"
	@echo "              RE-OPTIMISATION. two 30-step descents from the shipped"
	@echo "              genome, identical but for the mesh. ~3.5 h. exits 0"
	@echo "make filletkt what \`Kt * agg\` is worth on a FILLETED mesh — §93's"
	@echo "              Condition B. reads six committed artifacts and solves"
	@echo "              nothing: the corner census (which singularity does the"
	@echo "              fillet actually remove), the divergence ladders, and the"
	@echo "              objective's modelled peak against the fillet surface's"
	@echo "              own converged one — the comparison an unfilleted mesh"
	@echo "              cannot host, because it has no measured side. exits 0"
	@echo "make filletpnorm the region-restricted p-norm over the fillet arc —"
	@echo "              §94 items 1 and 2. sweeps the exponent AND the tube"
	@echo "              radius, because the region's own quadrature is what"
	@echo "              decides whether an exponent means anything, and"
	@echo "              measures what an indicator region weight costs at a"
	@echo "              gene value where a Gauss point crosses the boundary."
	@echo "              ~4 min, eight solves. exits 0"
	@echo "make filletpnormbox the same (r, p) sweep across §82's 32 held-out"
	@echo "              genomes, not just the two named ones — is 16 a constant"
	@echo "              of the objective or a property of two designs. ~13 min,"
	@echo "              84 solves. exits 0"
	@echo "make filletconda Condition A — the convergence ladder at \`b029622\`,"
	@echo "              SVK, eight phases, coarse/medium/fine, both meshes: does"
	@echo "              the filleted mesh stay the converged one and does the"
	@echo "              admissibility disagreement (1.0112 / 0.7954) survive"
	@echo "              refinement. slow — hours, dominated by the filleted"
	@echo "              mesh's own JIT trace at each rung. exits 0"
	@echo "make filletwiring the last two things in front of the switch — §94's"
	@echo "              items 3 and 4. the \`P_c\` exclusion argued rather than"
	@echo "              assumed (its stated reason names a feature the mesh has"
	@echo "              not had since 2026-08-18, and the hub corner IS the"
	@echo "              part's, to 0.008 deg), and stress_margin's exchange"
	@echo "              rate re-derived — the weight is invariant, the"
	@echo "              REFERENCE POINT is not. ~3 s, no solve. exits 0"
	@echo "make triblock the rim tri-block, BUILT — §51's probe measured. the"
	@echo "              faithful rim's junction is a TRIANGLE; the three-quad"
	@echo "              Y-partition meshes at 0.626 against the quad's 0.0082,"
	@echo "              12 blocks and 17 whole-edge seams; plus the interior"
	@echo "              point re-derived against the gene box, and the CURVED"
	@echo "              Y — one genome refuses all of it; plus the REFUSAL search,"
	@echo "              the box drawn out to 64 genomes, and a draw CONDITIONED on"
	@echo "              arc span — 22 of 40 in the band refuse, against 1 of 64"
	@echo "              uniform. ~670 s"
	@echo "make trirule  the tri-block's fold rule, CALIBRATED — a fit-freeze-"
	@echo "              score-swap hold-out protocol on section 73's own"
	@echo "              subject, curved-Y refusal on the arc-span band. no"
	@echo "              threshold published in-sample; a held-out score,"
	@echo "              including an embarrassing one, is a finding. ~15 min"
	@echo "make tribend  a bend that is a FUNCTION of the genome, fit against"
	@echo "              a constant under the same discipline, at w held"
	@echo "              fixed to the tri-block's own cell. whether genome-"
	@echo "              dependence helps at all is reported either way. ~1 min"
	@echo "make reds-hub-fillet  FILLET_PLAN Step 3's ACCEPTANCE TEST, and the one"
	@echo "              the whole fillet arc was aimed at: the R_hub sweep on a"
	@echo "              FILLETED mesh under SVK. It stops being bit-identical —"
	@echo "              11 distinct values of 14 rows against the control's ONE —"
	@echo "              §14's hub-share direction survives, and the Kt term the"
	@echo "              objective prices R_hub through goes EXACTLY flat above"
	@echo "              its cap while the wheel keeps stiffening. ~80 s (§75)"
	@echo "make reds-hub-fillet-rungs  the hub-share LADDER on the filleted mesh —"
	@echo "              the gate's own quantity, which no target could produce"
	@echo "              until §109. five rungs x two genomes, linear so the only"
	@echo "              difference from reds-hub's ladder is the mesh. ~35 s"
	@echo "make stage3-resume TRAJ=f.json   recover a killed descent's best iterate"
	@echo "              into a genome file --genome can read, and print the restart"
	@echo "              command. --best-out is only written when a descent COMPLETES,"
	@echo "              so a kill leaves a full trajectory and nothing to restart from."
	@echo "              WARM restart: position only, Adam's moments start from zero"
	@echo "make boundarywaste  BOUNDARY_PLAN Step 0: does defect 5's wasted-descent"
	@echo "              ratio generalise across the 25 committed Stage-3 runs?"
	@echo "              solves nothing — reads each run's own steps array. Four"
	@echo "              of 25 bite, all medium+SVK, 17.5-42.8% each (§112)"
	@echo "make fillet   at what radius does the filleted spoke block fold? sweeps"
	@echo "              both junctions under three criteria — the two that"
	@echo "              FILLET_PLAN.md's PART 3 and PART 5 disagreed by 20x on,"
	@echo "              and detJ at the Gauss points. 20 s, geometry only (§44)"
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
# roughly (steps x phases x 18.9 s) at `coarse` — see study_stage3.py's S10.
#
# 18.9 s IS MEASURED ON THE FILLETED MESH (2026-09-03, S10 re-run after PLAN.md §103's
# switch): 151.42 s for a full 8-phase evaluation, 50.47 h projected for 300 steps x 4
# starts, 43.4 GiB peak RSS in one process.  The `0.7 s` this line carried from 2026-07-27
# never matched ANY S10 reading — the oldest on record is 18.05 s (2026-07-29) — so it
# priced the run 27x under from the day it was written.  Note `--steps` defaults to 60,
# not the 300 the S10 projection is quoted at.
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

# KINEMATICS_PLAN.md step 1.  DOES `linear` RANK DESIGNS THE WAY SVK DOES?
#
# The 22.75% correction at service load (§14, re-measured by §31 and by KINEMATICS_PLAN
# step 0a) condemns linear as a REPORTING model.  It does not, on its own, condemn it as a
# SEARCH model: an optimizer needs the right ordering and the right descent direction, not
# the right absolute deflection.  This scores every distinct committed genome under BOTH
# kinematics with no optimizer and asks that question against the three conditions
# KINEMATICS_PLAN step 0c registered BEFORE the run — argmin identity, Spearman rho, and
# the gradient cosine in normalized gene space.
#
# NOT IN `studies`, for the reason `svk`, `m8bi5`, `m9buck` and `hubcap` are not: it
# measures THE WHEEL, NOT THE COMMIT.  KINRANK_WORKERS is the memory cap and nothing else
# sizes it, exactly as SVK_WORKERS.
#
# `coarse` is the default rung and it is a choice, not a saving: §14 measured the GNL
# correction converged by `coarse` and mesh-independent to three digits on both the shipped
# and the GA/beam genome, so the quantity this driver ranks on does not move by going to
# `medium` — and `medium` costs ~5x.  Re-run with KINRANK_CONFIG=medium to check that.
KINRANK_CONFIG ?= coarse
KINRANK_WORKERS ?= 8
KINRANK_OUT ?= study_kinematics_rank.json

kinrank:
	$(PY_OPT) -u studies/study_kinematics_rank.py --config $(KINRANK_CONFIG) \
	    --workers $(KINRANK_WORKERS) --out $(KINRANK_OUT)

# CONTACT_PLAN.md step 2 — one cell of the patch-resolution matrix.
#
# `study_contact.py` is ALSO in `studies` and that full invocation is the M6 gate; this
# target is the opposite kind of run, which is why it is separate and why it refuses to
# write `study_contact.json`.  It runs ONE section (`patch`) on ONE genome under ONE
# strain measure, because the question is about the wheel and the mesh rather than about
# the commit, and the other six sections cost hours at `medium` to answer nothing that is
# being asked.  Same reason `svk`, `m8bi5`, `m9buck` and `hubcap` are not in `studies`.
#
# Exists at all so the five pinned env vars above reach the run: they are exported by
# `make` and nothing else sets them.  Run each cell sequentially and capped —
#
#   systemd-run --user --unit=contact-s2 -p MemoryMax=20G --collect \
#       make contact CONTACT_GENOME=best_solution.json CONTACT_KIN=svk \
#            CONTACT_OUT=study_contact_e126cc3_svk.json
#
CONTACT_GENOME ?= best_solution.json
CONTACT_KIN ?= linear
CONTACT_SECTIONS ?= patch
CONTACT_OUT ?= study_contact_step2.json

contact:
	$(PY_OPT) -u studies/study_contact.py --genome $(CONTACT_GENOME) \
	    --kinematics $(CONTACT_KIN) --sections $(CONTACT_SECTIONS) \
	    --no-plot --out $(CONTACT_OUT)

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

# DEFECT8_PLAN.md step 4 / PLAN.md §23 successor 1.  THE PRODUCTION DESCENT UNDER THE
# KNEE'D `stress_margin`.
#
# EVERY KNOB IS §19'S, DELIBERATELY: `medium`, SVK, 100 steps, uniform 8-phase, seed 0,
# 4 workers, `--fidelity-check-every 25 --fidelity-check-config coarse`, from the shipped
# genome, `--min-wall 1.2`.  §19's own run (`stage3_margin_medium.json`, 101 objective
# calls, 6 h 20 m) is therefore an exact control and the ONLY difference between the two
# is the objective — `w * util**2` there, `soft_barrier(util_j - 0.80)` here.  That is the
# same discipline `buildcap` above applies to `svk-medium`, and for the same reason: a run
# that changes two things measures neither.
#
# FROM THE SHIPPED GENOME, not from a probe iterate.  `e126cc3` sits at util 0.780 against
# a knee at 0.800, so under the new objective its margin term is inert and `mass` has
# nothing opposing it until utilisation climbs back to the knee.  Step 0 of this run IS
# the shipped genome re-scored under the new objective, which is what makes the step-0 vs
# selected-iterate table apples to apples the way §19's was.
#
# DISTINCT --out AND --best-out, load-bearing for the reason prod9/prod10's and buildcap's
# are: `stage3_margin_medium.json` is the control and clobbering it would destroy it.
#
# ~226 s/step on the three `medium`/SVK/100-step runs this mirrors, so ~6.3 h.  Launch it
# capped and detached, exactly as prod9/prod10 and svk-shipped —
#
#   systemd-run --user --unit=wheel-knee -p MemoryMax=32G --collect \
#       --working-directory=$$PWD /usr/bin/make knee
#
# 32G, NOT the 16G every block above uses, and this is measured rather than inherited.  The
# 2026-08-13 run was launched at 16G and sat at the ceiling: `memory.current` 15.3 GiB,
# `memory.events` max = 2936 forced direct reclaims, `oom_kill` 0.  It survived and held the
# control's pace (224-239 s/step against §19's 236/191/227), so nothing in this file's timings
# moves — but a `medium` SVK descent WANTS more than 16 GiB and spends five hours one
# allocation from the kill switch.  The cap was raised to 32G on the live unit with
# `systemctl --user set-property`.  The 16G above it was sized against a 31 GB box; this one
# now reports 61 GiB total with 49 free, so the "two descents do not fit" arithmetic that
# every other block here inherits is worth re-deriving before it is trusted again.
KNEE_STEPS ?= 100
KNEE_GENOME ?= best_solution.json
KNEE_OUT ?= stage3_knee_medium.json
KNEE_BEST ?= stage3_knee_best_medium.json

knee:
	$(PY_OPT) -u src/wheel_stage3.py --start best \
	    --genome $(KNEE_GENOME) \
	    --config medium --kinematics svk --min-wall $(SVK_MIN_WALL) \
	    --steps $(KNEE_STEPS) --workers $(SVK_DESCENT_WORKERS) \
	    --phase-scheme uniform \
	    --fidelity-check-every 25 --fidelity-check-config coarse \
	    --out $(KNEE_OUT) --best-out $(KNEE_BEST)

# The milestone gates.  These are not tests — they produce measured reports whose
# numbers are quoted in CLAUDE.md — but they do exit nonzero when a gate fails, so
# they are safe to run in CI.
#
# WHAT A DRIVER HERE MAY EXIT NONZERO FOR — PLAN.md §33, 2026-08-16.
# A driver stops this recipe only when ITS OWN MEASUREMENT IS NOT TRUSTWORTHY: a solver
# identity violated, a mesh degenerate, a gradient that disagrees with finite differences.
# A driver must NOT stop the recipe for a CHARACTERISATION FINDING — a true, reproduced,
# deliberately-held statement about the design being measured.  The two look identical
# from `make`, which sees only an exit status, and conflating them cost this tree the
# whole recipe for ten days:
#
#   `study_gnl.py` exited 1 because the shipped 1.2 mm wheel is more geometrically
#   nonlinear than a pre-registered gate — a finding four arcs have deliberately kept
#   red — and `make` stopped at line 5 of 9.  study_contact, study_gradient,
#   study_objective and study_stage3 were unreachable from 2026-08-06 to 2026-08-16,
#   which is why §15's stale study_gradient.json was never refreshed.  Every artifact
#   this recipe writes carried the same 2026-08-03 date, INCLUDING the four drivers
#   before study_gnl, which is what proved it had not run at all rather than aborting
#   partway.  See the SEMANTICS block in studies/study_gnl.py.
#
# Before adding a hard stop to a driver, ask which of the two it is.  Report the finding
# loudly in the verdict block either way — a gate that goes quiet is worse than one that
# stops the build.
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

# SVK_PLAN.md's closing item / PLAN.md §26 successor #2 — the deflection gate's standing.
#
# The ±0.3% gate is satisfiable at exactly ONE rung (SVK_PLAN step 6: the coarse-converged
# answer read +1.65% at medium, the medium-converged answer -1.71% at coarse), so the
# number a design is judged against moves when the rung moves.  This runs the ladder the
# gate's OWN QoI — `axle_drop_mean_mm`, 8-phase, both kinematics — and extrapolates it, so
# the gate can be stated against a mesh-independent value instead of a rung.
#
# NOT in `make studies`, for the same reason `svk` is not: it measures THE WHEEL, NOT THE
# COMMIT.
#
# COST, MEASURED, because it was first estimated at "~11 min, ~2.5 GB" from bare
# `solve_wheel` timings and both numbers were wrong by an order of magnitude: the
# objective does far more per phase than one solve (`medium`/svk took 626 s against a
# predicted 144), and with --workers 0 all eight phase meshes are resident at once.
#
#   RUN 2026-08-14, --workers 0, smoke+coarse+medium+fine, both kinematics:
#     wall 5692 s (1 h 35 m)      peak RSS 20.6 GB      box 61 GB, 35 free, no swap
#     per rung (linear/svk s):  smoke 204/22   coarse 310/173   medium 633/626
#                               fine 1758/1967
#
# So it fits on this box uncapped but it is NOT a cheap target, and `fine` is ~3x medium.
# Use --reanalyse to redo the arithmetic on a saved report for free; an analysis bug
# should not cost 95 minutes, and TWO already have — the Roache index swap, and then the
# cell size being measured on `wheel_mesh`'s spoke block instead of the `wheel_wheel` mesh
# the QoI is solved on (PLAN §29).  Both were fixed by reanalysis alone, no FEA:
#   $(PY_OPT) studies/study_deflection_gci.py --reanalyse studies/study_deflection_gci.json
# `tests/test_deflection_gci.py` now pins the arithmetic and the mesh the counts come from.
GCI_GENOME ?= best_solution.json
GCI_LADDER ?= smoke,coarse,medium,fine
GCI_WORKERS ?= 0
GCI_OUT ?= studies/study_deflection_gci.json

gci:
	$(PY_OPT) -u studies/study_deflection_gci.py --genome $(GCI_GENOME) \
	    --ladder $(GCI_LADDER) --workers $(GCI_WORKERS) --out $(GCI_OUT)

# Does the junction corner actually carry a stress singularity?  PLAN §30.
#
# CHEAP, and that is the point — one LINEAR single-phase solve per rung instead of `gci`'s
# eight phases under both kinematics.  MEASURED 2026-08-15, whole ladder to `fine`:
# wall 8.5 s, peak RSS 1.56 GB, against `gci`'s 5692 s and 20.6 GB.  It answers the
# question §29 spent 95 minutes failing to answer, because it measures the field AT the
# corner instead of inferring a mechanism from the convergence order of a global
# functional.  Nothing about the expensive study was needed to settle it.
CORNER_GENOME ?= best_solution.json
CORNER_LADDER ?= smoke,coarse,medium,fine
CORNER_OUT ?= studies/study_corner_singularity.json

corner:
	$(PY_OPT) -u studies/study_corner_singularity.py --genome $(CORNER_GENOME) \
	    --ladder $(CORNER_LADDER) --out $(CORNER_OUT)

# THE SAME LADDER ON A FILLETED MESH.  FILLET_PLAN.md Step 2, reachable since PART 11.
#
# ~22 s and it is the same driver — one flag — which is the point: a filleted "before and
# after" measured by two scripts is two instruments, and this arc has already been bitten
# once by exactly that (PART 6, two recorded fold tables disagreeing 20x with neither
# criterion written down).  `--fillet genome` takes genes 12 and 13.
#
# `--continuity coarse` is NOT decoration.  The filleted ladder reports an axle drop 38%
# below the unfilleted one, and one ladder cannot tell the fillet's stiffness from a
# different model.  The control drives the radius pair toward zero and asks the filleted
# blocking to reproduce the unfilleted wheel; it does, to -0.17% at R = 0.05 mm.  Thirteen
# extra `coarse` solves, ~8 s of the 22.
#
# SCOPE, WHICH IS PART 10's AND HAS NOT MOVED: `fillet=` is a MEASUREMENT INSTRUMENT for
# one genome.  6 of 16 feasible genomes refuse it at their own radii.  Nothing on this
# recipe may be read as a licence to wire it into `wheel_objective` or the GA.
CORNER_FILLET ?= genome
CORNER_FILLET_CONTINUITY ?= coarse
CORNER_FILLET_OUT ?= studies/study_corner_singularity_fillet.json

corner-fillet:
	$(PY_OPT) -u studies/study_corner_singularity.py --genome $(CORNER_GENOME) \
	    --ladder $(CORNER_LADDER) --fillet $(CORNER_FILLET) \
	    --continuity $(CORNER_FILLET_CONTINUITY) --profiles --out $(CORNER_FILLET_OUT)

# ---------------------------------------------------------------------------
# DOES THE MESH HAVE THE CORNERS THE PART HAS?  (UNCAP_PLAN.md, PLAN §34)
# ---------------------------------------------------------------------------
# ~4 s, GEOMETRY ONLY — no field is solved, so this is not on the `studies` recipe's
# critical path and can be re-run by anyone who doubts a number in UNCAP_PLAN.md.
#
# It reproduces `wheel_step_export._embed` in numpy so it runs in the OPT env with no OCC,
# and it self-checks two ways: the ring-crossing count must come out 24 and 24 against the
# shipped manifest's `hub_edges`/`rim_edges`, and the mesh's four corner wedges must
# reproduce `make corner`'s independently measured ones to under 1 deg.  It EXITS NONZERO
# only on that reconstruction check failing — never on a characterisation finding about
# the wheel.  See the note above `studies:` for why that distinction is load-bearing.
JUNCTION_GENOME ?= best_solution.json
JUNCTION_CONFIG ?= coarse
JUNCTION_OUT ?= studies/study_junction_agreement.json

junction:
	$(PY_OPT) -u studies/study_junction_agreement.py --genome $(JUNCTION_GENOME) \
	    --config $(JUNCTION_CONFIG) --out $(JUNCTION_OUT)

# ---------------------------------------------------------------------------
# AT WHAT RADIUS DOES THE FILLETED SPOKE BLOCK FOLD?  (FILLET_PLAN.md, PLAN §44)
# ---------------------------------------------------------------------------
# ~20 s, GEOMETRY AND JACOBIANS ONLY — no field is solved, so like `junction` this is not
# on the `studies` recipe's critical path and anyone who doubts a fillet number can re-run
# it.  It is the apparatus FILLET_PLAN.md's PART 5 said had to exist before either route
# out of PART 3 is attempted: the two recorded "largest surviving radius" tables (PART 3's
# 0.20/0.10 and PART 5's 4.00/3.00/0.40) disagreed by 10-20x with neither criterion
# written down and no script of either surviving.
#
# It reproduces BOTH tables from one sweep and shows they are different criteria — PART 3
# counted mixed-sign cells in the spoke block, PART 5 asked whether `build_wheel` raises —
# and it adds the criterion that decides which to believe: `det J` at the Gauss points the
# FE assembly integrates.  It EXITS NONZERO only on a self-check failing (the two
# zero-radius controls, or either table no longer reproducing), never on a
# characterisation finding about the fillet.
FILLET_OUT ?= study_fillet_fold.json

fillet:
	$(PY_OPT) -u studies/study_fillet_fold.py --out $(FILLET_OUT)

# ---------------------------------------------------------------------------
# CAN THE FILLET BE A BLOCK, AND CAN THE SECTOR BE BLOCKED AROUND IT?
# (FILLET_PLAN.md PART 9 and PART 10)
# ---------------------------------------------------------------------------
# ~85 s, GEOMETRY AND JACOBIANS ONLY, and like `fillet` and `junction` it is off the
# `studies` critical path.  `make fillet` settled what "valid" MEANS for a filleted mesh
# (det J at the Gauss points); this asks the question in front of it — whether either of
# PART 3's two routes has a REGION that can be a block — before a week is spent building
# one.  Same discipline as PART 7 and PART 8: re-check the premise before spending.
#
# It reports three things and exits nonzero on none of them: that the region PART 3 named
# has TWO CUSPS (0.0000 deg at B, 0.42-0.60 at A) so route 1 as written is not a block;
# that route 2's failing angle is between two BOUNDARY curves and an elliptic interior
# solve leaves it bit-identical; and that a BOUNDARY-LAYER block whose corners are off
# both tangent points meshes at every radius in the gene box, min scaled Jacobian 0.91+.
#
# PART 10 added the half that a single block cannot answer: the WHOLE filleted sector,
# ELEVEN blocks and FOURTEEN seams, every seam whole-edge.  The fillet block's inner edge
# crosses the ring circle, so it has two partners unless it is split at the crossing, and
# the ring blocks it lands in have to close as quads whose node counts agree.  Measured
# at `coarse` and `medium` across the admissible gene box: 48/48 cells valid AND closed,
# worst min scaled Jacobian 0.357 against the unfilleted sector's 0.783 and
# MIN_SJ_TARGET's 0.2, worst seam gap 1.4e-14 mm.  Two prices come with it — the ring's
# radial node count becomes `n_thick` rather than `n_collar_r`/`n_rim_r`, and `R_hub` is
# now bounded at 3.130 mm by the SECTOR (the tangent point reaches the next sector's
# corner) rather than by the block, which is still clean at 4.00.
#
#
# AND IT NAMES ITS OWN SCOPE, because the radius box is not the gene box.  Sixteen freshly
# drawn feasible genomes, four per flank orientation, close every seam — after they found
# the bug that the sector-closing seam's `dk` follows the genome and not the constant +1 —
# but SIX REFUSE the fillet at their own shipped radii and only four of the ten that build
# clear the barrier.  So the blocking is fit for STEP 2, which needs one filleted mesh at
# one genome, and is NOT yet fit for the optimizer, which sweeps genomes.
#
# It EXITS NONZERO only on a self-check: the controls, the exactness of the cusp at B,
# the route-2 invariance, that every seam of the filleted sector CLOSES whole-edge at the
# shipped genome AND at every flank orientation, and that the unfilleted sector is still
# clean.  A block's min scaled Jacobian is a characterisation finding and is reported,
# never gated.
FILLETBLOCK_OUT ?= study_fillet_block.json

filletblock:
	$(PY_OPT) -u studies/study_fillet_block.py --out $(FILLETBLOCK_OUT)

# ---------------------------------------------------------------------------
# WHAT THE FILLETED OBJECTIVE COSTS (PLAN.md §89 ranked successor 1)
# ---------------------------------------------------------------------------
# THE NUMBER §88 QUOTED AND NOBODY MEASURED.  §88's ranking item 2 said the filleted mesh
# is "2-3x the cost of the unfilleted one", and §89 found that nothing in this tree
# measures it -- the only adjacent number being the element counts at `coarse`, 5952
# against 4704, which is 1.27x and is not solve time.  With §48's mesh-validity clause
# retired at §89, that cost and the flat `Kt` surrogate (§75) are the whole of what stands
# between the fillet and the objective, so the cost half is measured here.
#
# Three altitudes, because "cost" is three quantities: the mesh build; one solve and one
# adjoint (`study_gradient`'s G10 method, extended to the mesh G10 never ran on); and ONE
# `wheel_objective.objective` evaluation at the optimizer's eight-phase stencil, which is
# the quantity the decision is about.  BOTH KINEMATICS -- `objective` called bare takes
# `wheel_fem`'s linear default while Stage 3 passes svk (§32), and the verdict reads the
# SVK row because that is what an optimizer step pays.
#
# Everything timed is POST-TRACE, and the trace is reported rather than discarded: the
# first evaluation at each setting pays a jit that Stage 3 pays once and amortises, and on
# the probe that sized this driver it was 117.7 s against a 19.8 s evaluation.
#
# It touches no `src/` module.  The meshes are built in the driver and handed to
# `objective(meshes=)`, which is the parameter `wheel_stage3.Evaluator` already uses, so
# `test_nothing_wires_the_fillet_into_the_objective` still holds and the scope gate stands.
#
# EXITS 0 ALWAYS.  A cost has no threshold to meet; §89 asked for the number.
FILLETCOST_OUT ?= study_fillet_cost.json

filletcost:
	$(PY_OPT) -u studies/study_fillet_cost.py --out $(FILLETCOST_OUT)

# ---------------------------------------------------------------------------
# WHERE THE 17x LIVES (PLAN.md §90 ranked successor 2)
# ---------------------------------------------------------------------------
# THE DIFFERENCE §90 MEASURED AND DELIBERATELY DID NOT ATTRIBUTE.  The filleted objective
# returns 671.66 against 38.79 at the shipped genome under svk, with |grad| 1179.53
# against 212.49, and §90 stopped at those two summary fields.  The attribution is not
# curiosity: if the mesh-validity barrier is most of the gap, the switch prices the
# INSTRUMENT and every committed loss number survives it; if the gap is in the T3 terms,
# the two meshes disagree about the WHEEL and every committed loss number is incomparable
# across the switch.  That is a promotion-shaped consequence and it wants knowing before
# the decision's last term (§75's flat `Kt` surrogate) arrives, not after.
#
# Two altitudes.  `tiers=("t1","t2")` is seconds and needs no solve and no kinematics --
# T1 reads the genes, T2 reads `mesh_coords`, and `min_sj` (§90's suspect) lives there.
# Then the whole eight-phase evaluation at BOTH kinematics, where the four solve-space
# terms appear and where the svk row reproduces §90's published pair.
#
# NOTHING IS TIMED HERE, SO NOTHING IS WARMED UP.  `filletcost` pays a warm-up call before
# every timed one; this driver reads VALUES, which a jit trace does not change, so the
# same four evaluations cost about half.  The trace is still paid (§90: 271.7 s
# unfilleted, 1122.0 s filleted) and `linear` runs first so it is the row that pays it.
#
# It touches no `src/` module -- the meshes are built in the driver and handed to
# `objective(meshes=)`, so the scope gate
# `test_nothing_wires_the_fillet_into_the_objective` still holds.
#
# EXITS 0 ALWAYS.  An attribution has no threshold to meet.  The one thing that WOULD be a
# failure -- `t1_identical` false, i.e. the two rows not being one objective on two
# meshes -- is printed and filed rather than raised.
FILLETTERMS_OUT ?= study_fillet_terms.json

filletterms:
	$(PY_OPT) -u studies/study_fillet_terms.py --out $(FILLETTERMS_OUT)

# ---------------------------------------------------------------------------
# WHAT THE SWITCH DOES TO THE OPTIMUM (PLAN.md §91 ranked successor 2)
# ---------------------------------------------------------------------------
# §91 attributed §90's 17x and it is NOT the barrier: all nine barrier terms are exactly
# 0.0 on both meshes, and 99.498% of the gap is `deflection` -- mean axle drop 1.9011 mm
# unfilleted against 0.9914 mm filleted, read against a FIXED 2.0 mm two-sided target.
# So the shipped genome is 4.945% under target on one mesh and 50.43% under on the other,
# and a design that far off a quadratic target is not sitting at that objective's minimum.
#
# THE QUESTION THIS ANSWERS IS THE PROMOTION-SHAPED ONE.  "The filleted objective scores
# the shipped genome at 671.66" is a fact about a number.  "The filleted objective's
# optimum is NOT the shipped genome" is a fact about the part, and it is what says whether
# the switch is followed by a Stage 3 re-run and a re-promotion or by nothing.
#
# TWO DESCENTS, ONE MESH APART.  Both from the shipped genome, both 30 cosine steps, both
# eight uniform phases under SVK at `coarse`, both with the same pinned flank orientation
# and the same box.  The CONTROL is `wheel_stage3.Evaluator` unchanged -- the objective as
# it ships, the one that promoted this genome -- so whatever IT moves in 30 steps is what
# "already converged, merely re-scored" looks like under this schedule.  That is why there
# is no threshold anywhere in the verdict: the criterion is the treatment arm measured
# against the control arm on two axes, distance walked and fraction of start loss removed.
#
# IT IS A LOWER BOUND AND SAYS SO.  30 steps is half of `wheel_stage3.DEFAULT_STEPS`, so
# what the filleted arm recovers bounds what re-optimising would recover from below.  The
# decision needs to know whether the optimum MOVES, not where it lands.
#
# THEN A 3x2 TABLE: the shipped genome and both endpoints, each read by BOTH objectives.
# Four cells are free (the arms' own first and last steps) and the two off-diagonal ones
# are separate evaluations at the same stencil, kernel and pinned orientation.  That is
# what prices the disagreement in both directions rather than only in the one the switch
# would take.
#
# ~3.6 h: §90 priced one filleted 8-phase svk evaluation at 183.6 s against 163.3 s, plus
# the one-time jit trace per mesh topology (271.7 s and 1122.0 s), plus the two
# cross-evaluations.  The control runs first so the filleted trace is paid by the second
# arm.
#
# It touches no `src/` module -- the filleted arm is a `studies/` subclass overriding one
# call -- so `test_nothing_wires_the_fillet_into_the_objective` still holds.  It writes no
# `best_solution.json` and neither endpoint is a candidate for anything.
#
# EXITS 0 ALWAYS.  Either answer is the answer; a driver that exited nonzero on
# `re_optimisation_not_re_score` would be asserting which one the decision wants to hear.
FILLETOPTIMUM_OUT ?= study_fillet_optimum.json

filletoptimum:
	$(PY_OPT) -u studies/study_fillet_optimum.py --out $(FILLETOPTIMUM_OUT)

# ---------------------------------------------------------------------------
# WHAT `Kt * agg` IS WORTH ON A FILLETED MESH  (PLAN.md §93's Condition B)
# ---------------------------------------------------------------------------
# ~10 s, AND IT SOLVES NOTHING.  Six committed artifacts in, one table out.  That is the
# point rather than a shortcut: the two sides of this comparison were measured by two
# different instruments on two different ladders, and re-measuring either one here would
# make it a third.
#
# §93 found, by reading `wheel_objective.py:1234` rather than the plan files, that
# `util_j = kt * agg / ALLOWABLE` applies a surrogate for a fillet on a mesh that has one.
# It also asserted two things about that finding from inspection, and this recipe checks
# both:
#
#   "on a filleted mesh the peak is finite and convergent"  — IT IS NOT.  §93 cited §52's
#   fillet-SURFACE numbers for a claim about the WHEEL, and §52's own headline says the
#   global maximum still sits on `rim:P_c` and still diverges.  Both `P_c` corners survive
#   the fillet at both genomes; both `P_t` corners do not.
#
#   "the direction is conservative, because Kt > 1"  — THAT IS ONE FACTOR OF A PRODUCT.
#   The other is a whole-wheel p=4 p-norm standing in for a local peak, and it errs
#   further the other way.  Measured against the fillet surface's own converged peak the
#   modelled peak is LOW, not high.
#
# The `--*-terms` inputs are `make filletterms` at the two genomes; the four `--*-corner-*`
# inputs are `make corner` and `make corner-fillet` at the two genomes.  `b029622` is the
# design the FILLETED objective descends to from the shipped genome (§92), filed at the
# repository root as `fillet_optimum_b029622.json` — NOT a promotion candidate, and its own
# `note` says so.
#
# EXITS 0 ALWAYS, for `filletterms`' reason: §93 asked what the term is worth, and a
# valuation has no threshold to meet.
FILLETKT_OUT ?= study_fillet_kt.json

filletkt:
	$(PY_OPT) -u studies/study_fillet_kt.py --out $(FILLETKT_OUT)

# ---------------------------------------------------------------------------
# THE REGION-RESTRICTED P-NORM OVER THE FILLET ARC  (PLAN.md §94, items 1 and 2)
# ---------------------------------------------------------------------------
# ~4 min, and every cell of a 5x9 sweep comes off eight solves — M8b-i.6 step 1's
# property, for its reason: the p-norm is a pure function of the converged displacement
# field, so re-solving per exponent would buy nothing.
#
# §94 proposed replacing `kt * agg` with the mesh's own reading of the fillet surface and
# put four things in front of it.  This recipe measures the first two, and the answer to
# the first one is not the shape §94 expected:
#
#   ITEM 1 — "sweep the exponent ... CHECK: the largest `p` whose observed order is still
#   ~2".  `p` IS NOT WHAT LIMITS THIS QUANTITY.  The region's own mass — a quadrature of
#   an integral over an analytically fixed region, with no displacement field in it at all
#   — already fails to converge at the tube radius §52's `arc_peak` uses, so no exponent
#   built on it can be quoted.  The sweep therefore runs over the RADIUS as well, because
#   that is the axis that decides whether the exponent means anything.
#
#   ITEM 2 — "needs a smooth weight in distance-to-arc, not an indicator".  MEASURED, and
#   the indicator's step is real: section C locates a gene value at which one Gauss point
#   crosses the tube boundary and samples across it at two step sizes a decade apart.
#
# `h` IS THE FILLET BLOCK'S, and the wheel's is carried beside it.  This measure is local
# to a block that refines 4 -> 8 -> 12 -> 16 elements a side while the wheel refines
# 1.617x then 1.556x by element count; those are different ladders and they give different
# orders off the same numbers.  `study_deflection_gci`'s H_DEFS block is the precedent and
# its warning is why: it drew `h` from the wrong mesh once and inflated every `p` by 1.25x.
#
# SCOPE: LINEAR, ONE PHASE — `study_corner_singularity`'s ladder, which is the one
# `arc_peak`'s numbers were taken on.  Nothing here touches `wheel_objective`.
#
# EXITS 0 ALWAYS, for `filletkt`'s reason: §94 asked what `p` the region measure supports,
# and "none at this junction, and here is the quantity that is actually limiting" is an
# answer rather than a failure.
FILLETPNORM_OUT ?= study_fillet_pnorm.json

filletpnorm:
	$(PY_OPT) -u studies/study_fillet_pnorm.py --out $(FILLETPNORM_OUT)

# ---------------------------------------------------------------------------
# (r, p) ACROSS §82's THIRTY-TWO HELD-OUT GENOMES  (PLAN.md §95/§99's successor 1)
# ---------------------------------------------------------------------------
# ~13 min, 84 solves (28 genomes x coarse/medium/fine — `smoke` dropped, it is
# excluded from every Richardson anyway).  §95 found r=0.45mm, p=16 the cross-cell
# answer at the shipped genome and at `b029622`, and said itself that two genomes
# is not a design space.  This reuses `study_fillet_pnorm`'s `build`/`verdict`
# UNCHANGED over the wider `genomes` list `studies/study_fillet_block.json`'s own
# held-out draw supplies, so the same function that answered "16, at two designs"
# now answers whether 16 survives the box.  26 of 32 held-out genomes build their
# own filleted sector at their own radii; the other 6 are excluded for §82's own
# stated reason, not this driver's.
#
# SCOPE: the same as `filletpnorm` — LINEAR, ONE PHASE.  Nothing here touches
# `wheel_objective`.
#
# EXITS 0 ALWAYS, for `filletpnorm`'s reason: whether 16 is a constant of the
# objective or a property of two designs is a measurement, not a threshold.
FILLETPNORMBOX_OUT ?= study_fillet_pnorm_box.json

filletpnormbox:
	$(PY_OPT) -u studies/study_fillet_pnorm_box.py --out $(FILLETPNORMBOX_OUT)

# ---------------------------------------------------------------------------
# CONDITION A — THE CONVERGENCE LADDER AT `b029622`, SVK, EIGHT PHASES
# (PLAN.md §93's sequence step 1 / §99's ranked successor 2)
# ---------------------------------------------------------------------------
# SLOW, AND ONE PROCESS PER CELL.  The whole-ladder, one-process form of this driver was
# killed three times short of `fine`, the last time ~1h25m in with a background-process
# memory cap reported as the cause — JAX does not release compiled executables between
# cells, so a long-lived process's baseline climbs across the ladder even though no
# single cell needs that much.  Six single-cell processes, each returning its memory to
# the OS on exit, is the one configuration this repo has observed to survive; `--finalise`
# then runs the analysis over the six merged cells and writes the verdict.  No cell needs
# `--workers`-style parallelism of its own, so this is six short recipe lines, not a loop.
#
# COST, MEASURED — RUN 2026-09-02, one cell per process:
#
#   cell                elem    wall       peak RSS
#   coarse  unfilleted   4704    503 s        9.1 GB
#   coarse  filleted     5952   1417 s       41.6 GB
#   medium  unfilleted  12288   1049 s       10.6 GB
#   medium  filleted    15552   2891 s       43.7 GB
#   fine    unfilleted  31200   2689 s       11.2 GB
#   fine    filleted    37632   3619 s       41.9 GB
#
# ~3h23m serial total.  The filleted arm's memory is dominated by a large, largely
# size-independent JIT-compile cost (41.6 GB at `coarse`, 41.9 GB at `fine` — 6.3x the
# elements for +0.7%), not a per-element one; the unfilleted arm stays under 12 GB
# throughout.  Both fit the 61 GB box with room to spare once run one cell at a time.
#
# §93's own check: does the filleted mesh's axle-drop spread stay materially smaller than
# the unfilleted mesh's (FILLET_PLAN STEP 1 PART 12's ordering, reproduced at this design
# and kernel), AND does the admissibility disagreement §94 measured at `coarse` alone
# (1.0112 unfilleted / 0.7954 filleted) survive refinement to `fine`.  If not, §93 says
# what happens: "this decision is reopened and nothing below happens."
#
# SCOPE: SVK, eight phases only — Condition A is not stated under linear.  Nothing here
# touches `wheel_objective`.
#
# EXITS 0 ALWAYS: Condition A failing is a live, meaningful outcome this arc already
# named, not a bug to signal nonzero about.
FILLETCONDA_OUT ?= study_fillet_condition_a.json

filletconda:
	$(PY_OPT) -u studies/study_fillet_condition_a.py --rung coarse --mesh unfilleted --out $(FILLETCONDA_OUT)
	$(PY_OPT) -u studies/study_fillet_condition_a.py --rung coarse --mesh filleted   --out $(FILLETCONDA_OUT)
	$(PY_OPT) -u studies/study_fillet_condition_a.py --rung medium --mesh unfilleted --out $(FILLETCONDA_OUT)
	$(PY_OPT) -u studies/study_fillet_condition_a.py --rung medium --mesh filleted   --out $(FILLETCONDA_OUT)
	$(PY_OPT) -u studies/study_fillet_condition_a.py --rung fine   --mesh unfilleted --out $(FILLETCONDA_OUT)
	$(PY_OPT) -u studies/study_fillet_condition_a.py --rung fine   --mesh filleted   --out $(FILLETCONDA_OUT)
	$(PY_OPT) -u studies/study_fillet_condition_a.py --finalise --out $(FILLETCONDA_OUT)

# ---------------------------------------------------------------------------
# THE LAST TWO THINGS IN FRONT OF THE SWITCH  (PLAN.md §94, items 3 and 4)
# ---------------------------------------------------------------------------
# ~3 s, AND IT SOLVES NOTHING.  Seven committed artifacts in, two arguments out.  The one
# thing it computes is a sector's BLOCKING, to count the fillet arcs the mesh actually
# makes rather than infer twelve from the spoke count.
#
# ITEM 3 asked for the `P_c` exclusion to be ARGUED rather than assumed, and offered a
# justification: "they are the END CAP's corners and the exported solid has no end cap".
# THAT IS THE PRE-2026-08-18 MESH.  `UNCAP_DEFAULT = (True, 1.0)` removed the cap from the
# mesh too (§38), and `wheel_wheel.py` says so at the paragraph that explains why it
# fillets only `P_t`.  Measured against `study_junction_agreement`'s numpy reconstruction
# of `_embed` — verified by its own crossing count, 24 and 24, against the manifest:
#
#   hub:P_c   0.008 deg of wedge from the part's second flank crossing   IS the part's
#   rim:P_c  50.612 deg from it                                          is NOT
#
# So both exclusions survive and NEITHER for §94's reason.  The rim's is fidelity.  The
# hub's is C1 — the peak diverges, which is the argument that made `Kt` necessary in the
# first place — and it leaves a real, singular corner of the real part unpriced, one the
# part FILLETS (24 of 24 edges at the full radius) where the mesh builds twelve arcs at
# `P_t` only.
#
# ITEM 4 asked for `stress_margin`'s exchange rate to be re-derived at the new scale.
# `w = mass_term / (2 (u - knee) u)` reproduces §23's published 328.49, and NOTHING IN IT
# KNOWS HOW `util` WAS COMPUTED: at a fixed reference the weight moves only with the mass
# term.  What cannot be done is take the reference from a design — under the replacement
# the shipped genome reads 1.1415 and `b029622` 1.6760, both outside the admissible
# [knee, wall] band §18 derived inside.  The whole curve is reported so the policy choice
# is visible; at the wall the weight is 89.21.
#
# AND THE KNEE, WHICH §94 DID NOT MENTION.  `MARGIN_KNEE_UTIL = 0.80`'s only stated
# evidence is "the shipped genome sits at 0.77952, essentially AT this knee".  Under the
# replacement it sits at 1.1415.
#
# EXITS 0 ALWAYS, for `filletkt`'s reason: §94 asked for an argument and a re-derivation,
# and neither has a threshold to meet.
FILLETWIRING_OUT ?= study_fillet_wiring.json

filletwiring:
	$(PY_OPT) -u studies/study_fillet_wiring.py --out $(FILLETWIRING_OUT)

# ---------------------------------------------------------------------------
# THE RIM TRI-BLOCK, BUILT (PLAN.md §37, §51 — UNCAP_PLAN Step 3)
# ---------------------------------------------------------------------------
# ~670 s, GEOMETRY AND JACOBIANS ONLY, and off the `studies` critical path like
# `filletblock`, `fillet` and `junction`.
#
# At the FAITHFUL rim -- `uncap` blend 0.0, what the exporter's `_embed` actually uses --
# the rim junction stops being a quadrilateral: the corner at `far_end` opens to 179.5
# deg, so the region is a curvilinear TRIANGLE and a four-sided block on it collapses to
# min scaled Jacobian 0.0072.  §37 named the three-quad Y-partition as the fix and
# shelved it on two clauses; §51 re-priced both with a scratch probe, said both were
# wrong, and filed the probe AS a probe with "do not quote these numbers until this
# exists".  This is what makes them quotable, and it supersedes them.
#
# BOTH OF §37's CLAUSES ARE RETIRED, MEASURED.  Clause 1's partial-edge seams: splitting
# the two shared neighbours instead cascades once and stops -- SEVEN blocks become TWELVE
# and SEVENTEEN seams, every one a whole edge of both blocks it names, closing at
# 7.1e-15 mm.  Clause 2's forced 1-element strip: real at §37's own free count (B = 8 ->
# 7x1, 3x1, 7x3, reproduced here as a self-check) and absent at three of the five B's the
# algebra admits, because B is the FREE side and its count was never inherited.
#
# AND IT MESHES, at the shipped genome, by a wide margin: min scaled Jacobian 0.6262 at
# `coarse` and 0.5816 at `medium`, against the un-partitioned block's 0.0082 / 0.0083 --
# a factor of 77 and 70 -- and 3.1x over MIN_SJ_TARGET, with the region's area conserved
# to 1e-5.  §51's probe said 0.25 and called it a floor; it was one.
#
# WHAT STOPS IT, AND IT IS A THIRD THING NEITHER §37 NOR §51 NAMED.  The faithful rim is
# NOT opt-in -- adopting it moves the mesh under every genome the search touches -- so the
# gene box is the measurement and one genome is not.  Sixteen freshly drawn feasible
# genomes, four per flank orientation: the shipped genome's own interior-point rule holds
# on 12/16 at `coarse` and 10/16 at `medium`, and re-sweeping the point per genome only
# reaches 15/16 and 12/16.  The ones it folds on are the WIDE weld arcs (16-41 deg against
# the shipped genome's 2.73), and at `coarse` the two ranges separate cleanly.  So the
# construction is PROVED and the RULE THAT PLACES ITS INTERIOR POINT is not.
#
# A Winslow solve on the interiors changes it by 0.000000: the worst corner is on a held
# boundary, so the number is set by where the Y's spokes GO.  A CURVED Y is the successor;
# a better smoother is not.
#
# It EXITS NONZERO only on a self-check: the two controls, that every declared seam
# closes, that there are twelve blocks and seventeen seams, that the partition covers the
# same region as the block it replaces, that the three cut neighbours are SLICES of what
# the tree builds today, and that the algebra reproduces §37's own arithmetic.  A block's
# min scaled Jacobian is a characterisation finding and is reported, never gated.
TRIBLOCK_OUT ?= study_tri_block.json

triblock:
	$(PY_OPT) -u studies/study_tri_block.py --out $(TRIBLOCK_OUT)

# ---------------------------------------------------------------------------
# THE TRI-BLOCK'S FOLD RULE, CALIBRATED (PLAN.md §73 successor 1)
# ---------------------------------------------------------------------------
# A fit-freeze-score-swap hold-out calibration of the conjunctive fold family
# `arc > t_wide OR (arc > t_conj AND min_wedge < t_wedge)` against section 73's own
# subject, curved-Y refusal (`curved_valid`) on `sweep_arc_span_band`'s draw -- four fresh
# band streams, ~200 s each.  Nonzero exit only on a self-check (stream disjointness, a
# band that missed its 40-genome target); a held-out accuracy, including an embarrassing
# one, is a finding and exits zero.  ~15 min.
TRIRULE_OUT ?= study_tri_rule.json

trirule:
	$(PY_OPT) -u studies/study_tri_rule.py --out $(TRIRULE_OUT)

# ---------------------------------------------------------------------------
# A BEND THAT IS A FUNCTION OF THE GENOME (PLAN.md §56 successor 2)
# ---------------------------------------------------------------------------
# Two one-parameter families -- a constant bend and bend = clip(k * bow_over_width, 0, 1)
# -- fit and scored under the same fit-freeze-score-swap discipline as `trirule`, at `w`
# held fixed to the tri-block's own published cell.  Answers whether genome-dependence
# helps at all; a "no" is reported in `self_checks["linear_beats_constant_on_holdout"]`
# rather than hidden.  Draws are the cheap uniform 16-genome box, not the band -- seconds,
# not minutes, per stream.
TRIBEND_OUT ?= study_tri_bend.json

tribend:
	$(PY_OPT) -u studies/study_tri_bend.py --out $(TRIBEND_OUT)

# ---------------------------------------------------------------------------
# THE REDS ARC (PLAN §31) — the two measurements that cleared the inherited reds
# ---------------------------------------------------------------------------
# CHEAP, both of them, and that matters: these replaced two test thresholds, so the
# evidence has to be re-runnable by whoever doubts them rather than quoted from a plan
# file.  `reds-ratio` is ~4 min wall across $(REDS_JOBS) processes; `reds-hub` is ~50 s.
#
# `reds-hub` MEASURED 2026-09-05 (PLAN §114): wall 44.3 s, PEAK RSS 3.05 GiB, in one process.
# The wall figure had been on file since §31 and the memory figure never had been, which made
# "CHEAP" a claim about time only -- the distinction that matters when something else already
# holds the box.  It is unfilleted, linear, single-phase, no gradient and no `wheel_objective`,
# so the 41.6-43.7 GB filleted-cell figure at :1124-1130 does NOT apply to it: that measures an
# 8-phase SVK objective cell WITH a gradient, and residency tracks phases held, not the compile
# (PLAN.md §105, :14798-14812).  `ultra` here is 70080 elements at 587208 dofs and costs 8.9 s.
#
# `reds-ratio` fans one cell per process because a cell is 2-6 s and the box has 24 cores,
# but every cell runs under the same five thread pins as `make test` (`studies/redsrun.sh`
# exports `wheel_pool.PINNED_ENV`) — a number quoted against a test's value has to be taken
# in the test's environment or it is a comparison between two differently-threaded runs.
REDS_JOBS ?= 10
REDS_CELLS ?= /tmp/reds-cells
REDS_RATIO_OUT ?= studies/study_reds_ratio_stability.json
REDS_HUB_OUT ?= studies/study_reds_hub_share.json

# Is `max/min over the drawn rows` a statistic a gate can sit on?  (It is not.)
# 109 cells: 20 seeds at each test's own n, an n sweep at seed 7, and the beam study in
# BOTH gene boxes — its statistic is a property of the box, not of the genome.
reds-ratio:
	@mkdir -p $(REDS_CELLS)
	@{ for s in $$(seq 0 19); do echo "beam $$s 6 2.0"; echo "beam $$s 6 1.2"; \
	     echo "gnl $$s 4 -"; done; \
	   for n in 12 24 48 96; do echo "beam 7 $$n 2.0"; echo "beam 7 $$n 1.2"; done; \
	   for n in 8 12 16 24 48; do echo "gnl 7 $$n -"; done; \
	   for s in $$(seq 0 9); do echo "beam $$s 12 2.0"; echo "beam $$s 24 2.0"; \
	     echo "gnl $$s 8 -"; echo "gnl $$s 16 -"; done; } \
	| xargs -P $(REDS_JOBS) -n 4 bash -c \
	    'w=""; [ "$$3" = "-" ] || w="--min-wall $$3"; \
	     studies/redsrun.sh studies/study_reds_ratio_stability.py --which $$0 \
	       --seed $$1 --n $$2 $$w 2>/dev/null \
	       > $(REDS_CELLS)/$$0_s$$1_n$$2_w$$3.json'
	@studies/redsrun.sh studies/study_reds_ratio_stability.py --which collect \
	    --glob "$(REDS_CELLS)/*.json" --out $(REDS_RATIO_OUT)

# PLAN §14 item 4b, finally measured: what moves the hub compliance share?
# --attribute names the gene that moves it, --rungs separates design from discretisation.
# `ultra` is built by the driver, not by `wheel_wheel.CONFIGS` — see the comment there for
# why it is not a rung the tree acquires.
#
# --sweep DID NOT KILL THE `R_hub` HYPOTHESIS, AND THIS COMMENT SAID IT DID (PLAN §75).
# On the unfilleted mesh every row of that sweep is bit-identical, because `R_hub` is gene
# 12 and moves no node — so the sweep could not answer §14 either way, and the driver's
# verdict line reported a falsification off an exact tie.  Use `reds-hub-fillet` below for
# the sweep that can.  This target is kept as the CONTROL: "it stopped being
# bit-identical" is only a finding against a run that was.
reds-hub:
	studies/redsrun.sh studies/study_reds_hub_share.py --sweep --attribute --rungs \
	    --config coarse --configs smoke,coarse,medium,fine,ultra --out $(notdir $(REDS_HUB_OUT))

# FILLET_PLAN Step 3's ACCEPTANCE TEST — the one the whole fillet arc has been aimed at
# (PLAN §75, FILLET_PLAN STEP 3 RECORD PART 1).  ~80 s at `coarse`.
#
# Meshes the junction fillets, so `R_hub` finally reaches the solve, and runs SVK rather
# than the linear default because FILLET_PLAN's cost section says Step 3 must not take that
# default silently.  It writes `sweep_filleted` alongside `reds-hub`'s `sweep` rather than
# over it.
#
# It reports 11 distinct values of 14 rows against the control's one; the hub share running
# 0.007755 -> 0.003703 over the feasible range, which is §14's direction; and the term the
# objective actually prices `R_hub` through going EXACTLY flat above its 0.6657 mm cap while
# the wheel keeps stiffening — 8.8% of axle drop over a span where the gradient is zero.
# Four rows clamp (PLAN §74) and are marked `*`; all four are infeasible anyway.
reds-hub-fillet:
	studies/redsrun.sh studies/study_reds_hub_share.py --sweep --fillet \
	    --config coarse --out $(notdir $(REDS_HUB_OUT))

# THE LADDER ON THE MESH THE OBJECTIVE SOLVES — the gate's own quantity, which no target
# could produce until 2026-09-04 (PLAN §106, §109).  `reds-hub` above runs `--rungs` on
# `build_wheel`'s unfilleted default and `reds-hub-fillet` runs `--sweep`, so the filleted
# ladder fell between them and §106 had to measure it by calling `shares()` by hand.
#
# LINEAR, NOT SVK, AND THAT IS DELIBERATE: `--fillet` implies SVK for the sweep because
# FILLET_PLAN's cost section says so, but this ladder's entire content is the plain rungs
# read against the filleted ones, and a kernel change on one side would confound the mesh
# change with a kinematics change.  The driver defaults the ladder to `linear` for that
# reason; it is named here anyway so the target states it rather than inheriting it.
#
# ~35 s, and it writes `rungs_filleted_linear` alongside `reds-hub`'s `rungs` rather than
# over it — the plain ladder is the control that licenses the filleted column.
reds-hub-fillet-rungs:
	studies/redsrun.sh studies/study_reds_hub_share.py --rungs --fillet \
	    --kinematics linear --configs smoke,coarse,medium,fine,ultra \
	    --out $(notdir $(REDS_HUB_OUT))

reds: reds-ratio reds-hub

# BOUNDARY_PLAN Step 0 (PLAN.md §112) — does defect 5's wasted-descent ratio generalise
# across the 25 committed Stage-3 runs, or was §19's run the outlier?  SOLVES NOTHING: it
# reads `terms.<name>.value` and `wall_s` off each run's own `steps` array, already on disk.
# Well under a second.  The answer is four of the 25, all `medium`+SVK, 17.5-42.8% each —
# not a property of the descent in general, so the arc stays at #8.
boundarywaste:
	$(PY_OPT) studies/study_boundary_waste.py

# THE RECOVERY PATH FOR A KILLED DESCENT (PLAN.md §114).  `wheel_stage3` persists its
# trajectory after every step, atomically, so that a long run survives a kill (`_persist`/
# `_write`, wheel_stage3.py:875-896) — but it writes the GENOME only after the whole descent
# returns (`:1299-1310`).  A run killed at step 299 of 300 therefore leaves a complete history
# and nothing promotable, and the history is not in the shape `--genome` reads: `load_genes`
# wants a TOP-LEVEL `genes` (`:980`) and a trajectory nests them under `best`/`final`.  This
# is that adapter.  SOLVES NOTHING: one JSON in, one JSON out, measured 0.11 s at 34 MB.
#
# A WARM RESTART, NOT A RESUME.  Adam's moments (`:589-590`), the secant `delta0`, `n_reject`
# and the decayed cosine lr (`:595`) are not recorded in a trajectory and cannot be recovered
# from one, so the restart recovers POSITION only.  The driver prints the restart command and
# names what it is assuming.
stage3-resume:
	@test -n "$(TRAJ)" || { echo "usage: make stage3-resume TRAJ=stage3_run.json"; exit 2; }
	$(PY_OPT) studies/stage3_resume_genome.py $(TRAJ)

# ---------------------------------------------------------------------------
# THE REQUIREMENTS LAYER (PLAN.md §97 — MBSE_PLAN Steps 0, 4, 5, 7)
# ---------------------------------------------------------------------------
# The arc that asks what the QUESTION was.  Three drivers and a front end, and only the
# third of them solves anything.
#
# `mbsebase` and `mbsecal` read committed artifacts and do arithmetic, in the idiom
# `filletkt` established.  They are seconds, they are off the `studies` critical path, and
# they exit nonzero on their own checks — `mbsebase` on the derivation reproducing the
# four shipped constants to 1e-12 and on the thermal curve being exactly 1.0 at 20 C,
# `mbsecal` on the points map being an identity at its own calibration point and on total
# exchange-rate pressure being invariant under any reallocation summing to 100.
#
# `mbsescore` is the only one that solves: seven evaluations with gradients at `coarse`
# under SVK at 8 phases — the five profiles plus the bit-identity pair.  It gates BOTH directions of the
# same criterion — the baseline profile compliant AND bit-identical to naming no
# requirements at all, and at least one other profile NON-COMPLIANT naming a binding
# requirement.  A verifier that cannot fail is not a verifier.
#
# `--kinematics svk` IS NOT OPTIONAL AND IS NOT A DEFAULT BEING RELIED ON.  `wheel_fem`'s
# kernel default is `linear` (§32) and eleven study drivers never mention it; the shipped
# genome was descended under SVK, and a loss from one strain measure is not comparable
# with a loss from the other (§14).  The driver's own default is `svk` and its gate guard
# refuses filing a `linear` run under the committed name.
MBSEBASE_OUT  ?= study_mbse_baseline.json
MBSECAL_OUT   ?= study_mbse_calibration.json
MBSESCORE_OUT ?= study_mbse_score.json
# No mission or priority flags: the front end's defaults ARE the shipped wheel's implied
# mission, and `--points` omitted leaves `DEFAULT_WEIGHTS` untouched rather than
# re-deriving the calibrated allocation and re-applying it a couple of ulp off.
MBSE_ARGS     ?=

mbsebase:
	$(PY_OPT) -u studies/study_mbse_baseline.py --out $(MBSEBASE_OUT)

mbsecal:
	$(PY_OPT) -u studies/study_mbse_calibration.py --out $(MBSECAL_OUT)

mbsescore:
	$(PY_OPT) -u studies/study_mbse_score.py --out $(MBSESCORE_OUT)

mbse:
	$(PY_OPT) -u src/wheel_mbse.py $(MBSE_ARGS)
