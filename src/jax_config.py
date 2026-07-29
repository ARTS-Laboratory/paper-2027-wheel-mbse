"""
=============================================================================
  JAX CONFIGURATION — import this FIRST, before any jax.numpy use
=============================================================================
    import jax_config  # noqa: F401
    import jax.numpy as jnp

float64 is not optional here.  JAX defaults to float32, and at float32 a Newton
solve cannot reach the 1e-10 residual this project's verification asks for, a
finite-difference gradient check has no plateau at all (the roundoff floor sits
above the discretization error at every step size), and the patch test's 1e-12
tolerance is below machine epsilon.  Every one of those failures reads as a bug in
the physics rather than a bug in the dtype.

`jax.config.update` must run before the first trace, which is why this lives in its
own module instead of at the top of `wheel_fem.py` — several modules need it and
only the first import does the work.

This module is in the JAX half of the import graph.  `wheel_fea.py` and
`wheel_geometry.py` must NOT import it: they are imported across the interpreter
boundary from the CadQuery environment, which has no jax at all
(`tests/test_import_hygiene.py` enforces that).
=============================================================================
"""

import jax

jax.config.update("jax_enable_x64", True)

# Fail loudly rather than silently producing float32 results if some other import
# got there first and disabled x64.
if jax.numpy.zeros(1).dtype != jax.numpy.float64:
    raise RuntimeError(
        "jax_enable_x64 did not take effect — something traced before this import. "
        "`import jax_config` must precede every other jax use."
    )
