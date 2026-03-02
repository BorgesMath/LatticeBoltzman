# poisson.py
import numpy as np
from numba import njit
from config.config import SOR_OMEGA # Correção aqui


@njit
def solve_poisson_magnetic(psi, chi_field):
    """
    Resolve a equação de Poisson para o potencial magnético escalar
    via Sucessiva Sobre-Relaxação (SOR).
    """
    ny, nx = psi.shape
    for _ in range(15):  # Iterações SOR internas por passo de tempo LBM
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                # Interpolação da permeabilidade magnética nas faces do volume de controle
                mu_E = 1.0 + 0.5 * (chi_field[y, x] + chi_field[y, x + 1])
                mu_W = 1.0 + 0.5 * (chi_field[y, x] + chi_field[y, x - 1])
                mu_N = 1.0 + 0.5 * (chi_field[y, x] + chi_field[y + 1, x])
                mu_S = 1.0 + 0.5 * (chi_field[y, x] + chi_field[y - 1, x])

                denom = mu_E + mu_W + mu_N + mu_S

                if denom > 1e-12:
                    psi_new = (mu_E * psi[y, x + 1] + mu_W * psi[y, x - 1] +
                               mu_N * psi[y + 1, x] + mu_S * psi[y - 1, x]) / denom

                    # Atualização SOR
                    psi[y, x] = (1.0 - SOR_OMEGA) * psi[y, x] + SOR_OMEGA * psi_new

        # Condições de Contorno (Neumann para H tangencial nulo nas paredes transversais)
        psi[0, :] = psi[1, :]
        psi[ny - 1, :] = psi[ny - 2, :]

    return psi