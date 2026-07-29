"""Put `src/` on the path for pytest AND for the subprocesses the tests spawn.

`pyproject.toml`'s `pythonpath` covers pytest's own imports, but three tests shell out to a
fresh interpreter — `test_import_hygiene` (a jax-free `import wheel_fea`), `test_cli`
(`python src/wheel_fea.py`) and `test_export_contract` (the CadQuery env cross-check).
Those children inherit `os.environ`, not pytest's config, so without the `PYTHONPATH` line
below a bare `pytest` would fail where `make test` passes — which is exactly the kind of
difference that gets diagnosed as a broken test rather than a broken path.
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
