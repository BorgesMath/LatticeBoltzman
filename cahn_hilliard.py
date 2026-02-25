# cahn_hilliard.py
import numpy as np
from numba import njit, prange
from config import BETA, KAPPA, DT_CH, M_MOBILITY


@njit(parallel=True)
def cahn_hilliard_substep(phi, u_x, u_y):
    """
    Resolve a equação de Cahn-Hilliard via discretização explícita com advecção Upwind.
    """
    ny, nx = phi.shape
    phi_next = np.zeros_like(phi)
    mu = np.zeros((ny, nx))

    # 1. Cálculo do Potencial Químico (Bulk + Gradiente)
    for y in prange(1, ny - 1):
        for x in range(1, nx - 1):
            lap_phi = phi[y, x + 1] + phi[y, x - 1] + phi[y + 1, x] + phi[y - 1, x] - 4.0 * phi[y, x]
            mu[y, x] = 4.0 * BETA * phi[y, x] * (phi[y, x] ** 2 - 1.0) - KAPPA * lap_phi

    # 2. Conservação e Transporte (Advecção + Difusão de Interface)
    for y in prange(1, ny - 1):
        for x in range(1, nx - 1):
            lap_mu = mu[y, x + 1] + mu[y, x - 1] + mu[y + 1, x] + mu[y - 1, x] - 4.0 * mu[y, x]
            ux, uy = u_x[y, x], u_y[y, x]

            # Advecção direcional (Upwind de 1ª ordem) para estabilidade
            dphi_dx = (phi[y, x] - phi[y, x - 1]) if ux > 0 else (phi[y, x + 1] - phi[y, x])
            dphi_dy = (phi[y, x] - phi[y - 1, x]) if uy > 0 else (phi[y + 1, x] - phi[y, x])

            phi_next[y, x] = phi[y, x] + DT_CH * (M_MOBILITY * lap_mu - (ux * dphi_dx + uy * dphi_dy))

    # 3. Condições de Contorno (Dirichlet na entrada, Neumann nas demais)
    phi_next[:, 0] = 1.0
    phi_next[:, -1] = phi_next[:, -2]

    return np.clip(phi_next, -1.0, 1.0)