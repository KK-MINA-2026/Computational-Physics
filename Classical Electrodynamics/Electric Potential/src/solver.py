import numpy as np
from numba import jit

@jit(nopython=True)
def compute_potential(potential, fixed_bool, n_iter):
    length = potential.shape[0]
    for _ in range(n_iter):
        for i in range(1, length - 1):
            for j in range(1, length - 1):
                if not fixed_bool[i][j]:
                    potential[i][j] = (
                        potential[i+1][j] +
                        potential[i-1][j] +
                        potential[i][j+1] +
                        potential[i][j-1]
                    ) / 4
    return potential


def compute_field(potential):
    E_x, E_y = np.gradient(-potential)
    E = np.sqrt(E_x**2 + E_y**2)
    return E_x, E_y, E