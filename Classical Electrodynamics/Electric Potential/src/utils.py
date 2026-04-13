import numpy as np

def create_grid(n=300):
    edge = np.linspace(-2, 2, n)
    xv, yv = np.meshgrid(edge, edge)
    return edge, xv, yv


def boundary_conditions(edge):
    upper_y = np.cos(np.pi * edge / 2)
    lower_y = edge**4
    upper_x = 1 /(np.e**(-1) - np.e) * (np.exp(edge) - np.e)
    lower_x = 0.5 * (edge**2 - edge)
    return upper_y, lower_y, upper_x, lower_x


def potential_block(x, y):
    return np.select(
        [(x > 0.5) & (x < 0.7) & (y > 0.5) & (y < 0.7)],
        [1],
        default=0
    )