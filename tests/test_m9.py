"""Fast M9 Phase 2 diagnostics checks."""

import os
import sys

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import wheel_adjoint as WA  # noqa: E402


def test_lowest_tangent_eigenpair_reports_a_converged_smallest_mode():
    matrix = sp.diags(np.arange(1.0, 21.0), format="csc")
    solve = spla.factorized(matrix)
    value, vector, diag = WA.lowest_tangent_eigenpair(matrix, solve, seed=0)

    assert value == pytest.approx(1.0, rel=1e-10)
    assert np.linalg.norm(vector) == pytest.approx(1.0)
    assert diag["converged"]
    assert diag["residual_rel"] < WA.LOBPCG_RESIDUAL_REL
    assert diag["iterations"] <= WA.LOBPCG_MAXITER
