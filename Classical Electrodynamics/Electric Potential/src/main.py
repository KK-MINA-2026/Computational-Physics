import numpy as np
import matplotlib.pyplot as plt
from src.solver import compute_potential, compute_field
from src.utils import create_grid, boundary_conditions, potential_block

edge, xv, yv = create_grid()

upper_y, lower_y, upper_x, lower_x = boundary_conditions(edge)

potential = np.zeros((len(edge), len(edge)))

potential[0, :] = lower_y
potential[-1, :] = upper_y
potential[:, 0] = lower_x
potential[:, -1] = upper_x

fixed = potential_block(xv, yv)
fixed_bool = fixed != 0
potential[fixed_bool] = fixed[fixed_bool]

potential = compute_potential(potential, fixed_bool, 10000)

E_x, E_y, E = compute_field(potential)

plt.contourf(xv, yv, potential, 40, cmap="coolwarm")
plt.colorbar()
plt.show()