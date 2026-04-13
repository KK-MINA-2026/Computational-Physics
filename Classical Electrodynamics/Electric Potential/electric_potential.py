import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import PillowWriter
from matplotlib import cm
from skimage import io
from skimage import color
import numba
from numba import jit

edge = np.linspace(-2, 2, 300)
upper_y = np.cos(np.pi * edge / 2)
lower_y = edge**4
upper_x = 1 /( np.e**(-1) - np.e ) * ( np.exp(edge) - np.e )
lower_x = 0.5 * (edge**2 - edge)

xv, yv = np.meshgrid(edge, edge)

@jit("f8[:,:](f8[:,:], i8)", nopython=True, nogil=True)
def compute_potential(potential, n_iter):
    length = len(potential[0])
    for n in range(n_iter):
        for i in range(1, length -1):
            for j in range(1, length -1):
                potential[i][j] = (potential[i+1][j] + potential[i-1][j] + potential[i][j+1] + potential[i][j-1]) / 4

    return potential

potential = np.zeros((len(edge), len(edge)))
potential[0, :] = lower_y
potential[-1, :] = upper_y
potential[:, 0] = lower_x
potential[:, -1] = upper_x
potential = compute_potential(potential, 30_000)

plt.contourf(xv, yv, potential, 100, cmap="coolwarm")
plt.colorbar()
plt.show()

def potential_block(x, y):
    return np.select([(x > 0.5) * (x < 0.7) * (y > 0.5) * (y < 0.7), 
                      (x <= 0.5) + (x >= 0.5) + (y <= 0.5) + (y >= 0.7)], 
                      [1, 0])

plt.figure(figsize=(3, 3))
plt.contourf(xv, yv, potential_block(xv, yv))

fixed = potential_block(xv, yv)
fixed_bool = fixed != 0

@jit("f8[:,:](f8[:,:], b1[:,:], i8)", nopython=True, nogil=True)
def compute_potential(potential,fixed_bool, n_iter):
    length = len(potential[0])
    for n in range(n_iter):
        for i in range(1, length -1):
            for j in range(1, length -1):
                if not fixed_bool[i][j]:
                    potential[i][j] = (potential[i+1][j] + potential[i-1][j] + potential[i][j+1] + potential[i][j-1]) / 4

    return potential

potential = np.zeros((len(edge), len(edge)))
potential[0, :] = lower_y
potential[-1, :] = upper_y
potential[:, 0] = lower_x
potential[:, -1] = upper_x
potential[fixed_bool] = fixed[fixed_bool]
potential = compute_potential(potential, fixed_bool, 1_000_000)

plt.figure(figsize=(6.5,5))
plt.contourf(xv, yv, potential, 40, cmap="coolwarm")
plt.colorbar()
plt.show()

E_x, E_y = np.gradient(-potential)
E = np.sqrt(E_x**2 + E_y**2)

plt.figure(figsize=(6.5,5))
plt.contourf(xv, yv, E, 400, cmap="coolwarm")
plt.colorbar()
plt.show()