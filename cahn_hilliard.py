import numpy as np
from numba import njit, prange
from config import BETA, KAPPA, DT_CH, M_MOBILITY, PERIODIC_Y


@njit(parallel=True)
def cahn_hilliard_substep(phi, u_x, u_y):
    ny, nx = phi.shape
    phi_next = np.zeros_like(phi)
    mu = np.zeros((ny, nx))

    # 1. Potencial Químico com Topologia Unificada
    for y in prange(ny):
        y_up = (y + 1) % ny if PERIODIC_Y else min(y + 1, ny - 1)
        y_dn = (y - 1) % ny if PERIODIC_Y else max(y - 1, 0)

        for x in range(1, nx - 1):
            lap_phi = phi[y, x + 1] + phi[y, x - 1] + phi[y_up, x] + phi[y_dn, x] - 4.0 * phi[y, x]
            mu[y, x] = 4.0 * BETA * phi[y, x] * (phi[y, x] ** 2 - 1.0) - KAPPA * lap_phi

    # Fronteiras de mu no eixo X (Neumann)
    for y in prange(ny):
        mu[y, 0] = mu[y, 1]
        mu[y, nx - 1] = mu[y, nx - 2]

    # 2. Conservação e Transporte
    for y in prange(ny):
        y_up = (y + 1) % ny if PERIODIC_Y else min(y + 1, ny - 1)
        y_dn = (y - 1) % ny if PERIODIC_Y else max(y - 1, 0)

        for x in range(1, nx - 1):
            lap_mu = mu[y, x + 1] + mu[y, x - 1] + mu[y_up, x] + mu[y_dn, x] - 4.0 * mu[y, x]
            ux, uy = u_x[y, x], u_y[y, x]

            dphi_dx = (phi[y, x] - phi[y, x - 1]) if ux > 0 else (phi[y, x + 1] - phi[y, x])
            dphi_dy = (phi[y, x] - phi[y_dn, x]) if uy > 0 else (phi[y_up, x] - phi[y, x])

            phi_next[y, x] = phi[y, x] + DT_CH * (M_MOBILITY * lap_mu - (ux * dphi_dx + uy * dphi_dy))

    # Fronteiras de phi no eixo X
    for y in prange(ny):
        phi_next[y, 0] = 1.0
        phi_next[y, nx - 1] = phi_next[y, nx - 2]

    return np.clip(phi_next, -1.0, 1.0)