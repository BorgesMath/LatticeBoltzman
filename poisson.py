import numpy as np
from numba import njit
from config import SOR_OMEGA, PERIODIC_Y


@njit
def solve_poisson_magnetic(psi, chi_field):
    ny, nx = psi.shape
    for _ in range(15):
        for y in range(ny):
            y_up = (y + 1) % ny if PERIODIC_Y else min(y + 1, ny - 1)
            y_dn = (y - 1) % ny if PERIODIC_Y else max(y - 1, 0)

            for x in range(1, nx - 1):
                mu_E = 1.0 + 0.5 * (chi_field[y, x] + chi_field[y, x + 1])
                mu_W = 1.0 + 0.5 * (chi_field[y, x] + chi_field[y, x - 1])
                mu_N = 1.0 + 0.5 * (chi_field[y, x] + chi_field[y_up, x])
                mu_S = 1.0 + 0.5 * (chi_field[y, x] + chi_field[y_dn, x])

                denom = mu_E + mu_W + mu_N + mu_S

                if denom > 1e-12:
                    psi_new = (mu_E * psi[y, x + 1] + mu_W * psi[y, x - 1] +
                               mu_N * psi[y_up, x] + mu_S * psi[y_dn, x]) / denom

                    psi[y, x] = (1.0 - SOR_OMEGA) * psi[y, x] + SOR_OMEGA * psi_new

        # Fronteiras de psi no eixo X
        for y in range(ny):
            psi[y, 0] = psi[y, 1]
            psi[y, nx - 1] = psi[y, nx - 2]

    return psi