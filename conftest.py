"""Put `src/` on the path for pytest AND for the subprocesses the tests spawn.

`pyproject.toml`'s `pythonpath` covers pytest's own imports, but three tests shell out to a
fresh interpreter — `test_import_hygiene` (a jax-free `import wheel_fea`), `test_cli`
(`python src/wheel_fea.py`) and `test_export_contract` (the CadQuery env cross-check).
Those children inherit `os.environ`, not pytest's config, so without the `PYTHONPATH` line
below a bare `pytest` would fail where `make test` passes — which is exactly the kind of
difference that gets diagnosed as a broken test rather than a broken path.

The thread pin below is the same argument for a different variable.  The Makefile exports
`wheel_pool.PINNED_ENV` so that a phase worker and the parent that compares against it
agree on how a reduction is associated, and `tests/test_pool.py` asserts pooled == serial
EXACTLY.  Under a bare `pytest` nothing had set them, so that test would have been
comparing two differently-threaded processes and failing for a reason with no connection
to the pool.

`XLA_FLAGS` is the entry that earns its place: measured, two plain serial runs of one
`coarse` adjoint in separate interpreters agree on every forward value to the bit and
disagree on the GRADIENT by 3.33e-16, because XLA's CPU thread pool does not associate its
reductions the same way twice.  Nothing in OMP/MKL/OPENBLAS/NUMEXPR touches that pool.

Set here rather than in a fixture because OpenBLAS, MKL and XLA read these once, at
import, and pytest has already imported numpy by the time any fixture runs.  Imported from
`wheel_pool` rather than retyped so the two lists cannot drift — it is stdlib+numpy and
pulls in no jax.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

_existing = os.environ.get("PYTHONPATH", "")
if SRC not in _existing.split(os.pathsep):
    os.environ["PYTHONPATH"] = (SRC + os.pathsep + _existing) if _existing else SRC

import wheel_pool  # noqa: E402  — needs SRC on sys.path, which the block above just did

for _name, _value in wheel_pool.PINNED_ENV.items():
    os.environ.setdefault(_name, _value)
