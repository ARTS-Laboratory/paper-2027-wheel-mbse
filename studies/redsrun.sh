#!/usr/bin/env bash
# Run a REDS measurement driver under exactly the environment `make test` runs tests in.
#
# The five thread pins are `wheel_pool.PINNED_ENV`; the Makefile exports them and
# `conftest.py` sets the same five, so a bare `pytest` and `make test` agree.  A
# measurement quoted against a test's value has to be taken in the same environment or the
# comparison is between two differently-threaded processes — see conftest.py's docstring.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# `src` and `studies` both, matching pyproject.toml's `pythonpath` — the study drivers are
# imported flat by the tests, so a scratch script that imports one has to resolve it the
# same way pytest does or it is measuring a different import.
export PYTHONPATH="$PWD/src:$PWD/studies${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
exec .venv-opt/bin/python "$@"
