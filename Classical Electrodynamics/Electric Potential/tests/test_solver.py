import numpy as np
from src.solver import compute_potential, compute_field
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_potential_convergence():
    grid = np.zeros((10, 10))
    fixed = np.zeros_like(grid, dtype=bool)


    grid[0, :] = 1

    result = compute_potential(grid.copy(), fixed, 200)

    assert not np.allclose(result, 0)


def test_fixed_points_unchanged():
    grid = np.zeros((10, 10))
    fixed = np.zeros_like(grid, dtype=bool)

    grid[5, 5] = 1
    fixed[5, 5] = True

    result = compute_potential(grid.copy(), fixed, 100)

    assert result[5, 5] == 1


def test_field_shape():
    grid = np.random.rand(20, 20)

    E_x, E_y, E = compute_field(grid)

    assert E.shape == grid.shape
    assert E_x.shape == grid.shape
    assert E_y.shape == grid.shape