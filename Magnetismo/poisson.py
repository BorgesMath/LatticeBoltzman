import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def solve_poisson_magnetic(psi, chi_field, h0, h_angle, sor_omega):
    ny, nx = psi.shape
    angle_rad = np.radians(h_angle)
    Hx_target = h0 * np.cos(angle_rad)
    Hy_target = h0 * np.sin(angle_rad)

    # Iterações do Solver SOR
    for _ in range(15):
        for y in prange(1, ny - 1):
            for x in range(1, nx - 1):
                mu_E = 1.0 + 0.5 * (chi_field[y, x] + chi_field[y, x + 1])
                mu_W = 1.0 + 0.5 * (chi_field[y, x] + chi_field[y, x - 1])
                mu_N = 1.0 + 0.5 * (chi_field[y, x] + chi_field[y + 1, x])
                mu_S = 1.0 + 0.5 * (chi_field[y, x] + chi_field[y - 1, x])

                denom = mu_E + mu_W + mu_N + mu_S

                if denom > 1e-12:
                    psi_new = (mu_E * psi[y, x + 1] + mu_W * psi[y, x - 1] +
                               mu_N * psi[y + 1, x] + mu_S * psi[y - 1, x]) / denom
                    psi[y, x] = (1.0 - sor_omega) * psi[y, x] + sor_omega * psi_new

        # Condições de Contorno de Dirichlet/Neumann para garantir campo uniforme distante da interface
        for y in prange(ny):
            psi[y, 0] = psi[y, 1] + Hx_target
            psi[y, nx - 1] = psi[y, nx - 2] - Hx_target

        for x in prange(nx):
            psi[0, x] = psi[1, x] + Hy_target
            psi[ny - 1, x] = psi[ny - 2, x] - Hy_target

    return psi