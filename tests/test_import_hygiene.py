"""The import contract that makes the two-interpreter pipeline work.

`wheel_step_export.py` runs in the CadQuery env and does
`from wheel_fea import generate_bezier_centerline, ...` (wheel_step_export.py:60-69).
That env has numpy and cadquery but NOT pygad, matplotlib, or jax.  `wheel_fea.py` keeps
itself importable there by lazy-importing its heavy dependencies inside `__main__`
(wheel_fea.py:948).

This is easy to break by accident — one convenience import at module scope and the STEP
exporter dies in an environment nobody runs tests in.  These tests are the tripwire.

They run in a subprocess because `sys.modules` is already polluted by the time pytest
has imported anything.
"""

import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Modules that must never be pulled in by `import wheel_fea`.
# jax is on the list pre-emptively: the FEA stages import it, and if the geometry kernel
# ever grows a top-level jax import the exporter breaks silently.
FORBIDDEN = ["pygad", "matplotlib", "jax"]


def _import_in_subprocess(module, forbidden):
    """Import `module` in a clean interpreter and report which forbidden modules it
    dragged in.  Returns (returncode, stdout, stderr)."""
    code = (
        "import sys\n"
        f"import {module}\n"
        f"leaked = [m for m in {forbidden!r} if m in sys.modules]\n"
        "print('LEAKED:' + ','.join(leaked))\n"
    )
    return subprocess.run([sys.executable, "-c", code], cwd=os.path.join(HERE, "src"),
                          capture_output=True, text=True)


def test_wheel_fea_imports_with_numpy_only():
    proc = _import_in_subprocess("wheel_fea", FORBIDDEN)
    assert proc.returncode == 0, f"import wheel_fea failed:\n{proc.stderr}"
    leaked = proc.stdout.strip().removeprefix("LEAKED:")
    assert leaked == "", (
        f"`import wheel_fea` pulled in {leaked}. The CadQuery env does not have these, "
        f"so wheel_step_export.py:60-69 would fail. Move the import into __main__ or "
        f"make it lazy."
    )


@pytest.mark.parametrize("name", [
    # exactly the symbols wheel_step_export.py:60-69 imports
    "generate_bezier_centerline", "thicken_3taper_curve", "stress_concentration_kt",
    "HUB_RADIUS_MM", "RIM_RADIUS_MM", "SPOKE_WIDTH_MM", "NUMBER_OF_SPOKES",
    "DENSITY_PLA",
])
def test_exporter_import_surface_is_intact(name):
    """Each name the STEP exporter reaches for must keep existing.  Parametrized so a
    failure names the exact symbol that went missing."""
    import wheel_fea
    assert hasattr(wheel_fea, name), (
        f"wheel_step_export.py imports `{name}` from wheel_fea; it is gone."
    )


def test_thicken_returns_edges_for_exporter():
    """`spoke_edges_global` (wheel_step_export.py:153) depends on the return_edges=True
    contract returning two hub->rim arrays of matching length.  A refactor that changed
    the return shape would only surface in the CAD env."""
    import numpy as np
    import wheel_fea

    curve, _ = wheel_fea.generate_bezier_centerline(5.9, 32.0, 17.5, 23.4,
                                                    19.5, 32.0, 33.5, 25.5)
    top, bot = wheel_fea.thicken_3taper_curve(curve, 2.4, 2.0, 2.0, 2.0,
                                              return_edges=True)
    assert top.shape == bot.shape == curve.shape
    assert np.all(np.isfinite(top)) and np.all(np.isfinite(bot))
