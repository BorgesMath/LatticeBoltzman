# Multifasico/cahn_hilliard_gpu.py
from numba import cuda
import math


@cuda.jit
def calc_mu_kernel(phi, mu, ny, nx, BETA, KAPPA):
    y, x = cuda.grid(2)
    if 0 < y < ny - 1 and 0 < x < nx - 1:
        lap_phi = phi[y, x + 1] + phi[y, x - 1] + phi[y + 1, x] + phi[y - 1, x] - 4.0 * phi[y, x]
        mu[y, x] = 4.0 * BETA * phi[y, x] * (phi[y, x] ** 2 - 1.0) - KAPPA * lap_phi


@cuda.jit
def ch_advance_kernel(phi, phi_next, mu, u_x, u_y, ny, nx, DT_CH, M_MOBILITY):
    y, x = cuda.grid(2)

    # Domínio interno
    if 0 < y < ny - 1 and 0 < x < nx - 1:
        lap_mu = mu[y, x + 1] + mu[y, x - 1] + mu[y + 1, x] + mu[y - 1, x] - 4.0 * mu[y, x]
        ux, uy = u_x[y, x], u_y[y, x]

        dphi_dx = 0.5 * (phi[y, x + 1] - phi[y, x - 1])
        dphi_dy = 0.5 * (phi[y + 1, x] - phi[y - 1, x])

        phi_next[y, x] = phi[y, x] + DT_CH * (M_MOBILITY * lap_mu - (ux * dphi_dx + uy * dphi_dy))

        # Limita os valores
        if phi_next[y, x] > 1.0: phi_next[y, x] = 1.0
        if phi_next[y, x] < -1.0: phi_next[y, x] = -1.0

    # Sincronização de contornos via threads (condições de borda)
    elif x == 0:  # Inlet Dirichlet
        phi_next[y, 0] = 1.0
    elif x == nx - 1:  # Outlet Neumann
        phi_next[y, nx - 1] = phi[y, nx - 2]