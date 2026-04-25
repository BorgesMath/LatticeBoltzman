import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def solve_poisson_magnetic(psi_tilde, chi_field, h0, h_angle, sor_omega):
    ny, nx = psi_tilde.shape
    angle_rad = np.radians(h_angle)

    # Decomposição do vetor do campo magnético exógeno
    Hx_fundo = h0 * np.cos(angle_rad)
    Hy_fundo = h0 * np.sin(angle_rad)

    # Iterações do Solver SOR para a PERTURBAÇÃO (Estritamente periódica em Y)
    for _ in range(15):
        for y in prange(ny):
            # Índices topológicos periódicos no eixo Y
            yp = (y + 1) % ny
            ym = (y - 1 + ny) % ny

            for x in range(1, nx - 1):
                # Interpolação aritmética das permeabilidades nas interfaces das células
                mu_E = 1.0 + 0.5 * (chi_field[y, x + 1] + chi_field[y, x])
                mu_W = 1.0 + 0.5 * (chi_field[y, x] + chi_field[y, x - 1])
                mu_N = 1.0 + 0.5 * (chi_field[yp, x] + chi_field[y, x])
                mu_S = 1.0 + 0.5 * (chi_field[ym, x] + chi_field[y, x])

                denom = mu_E + mu_W + mu_N + mu_S

                # Termo fonte dinâmico: Interação do campo macroscópico com o gradiente de fase
                rhs = Hx_fundo * (mu_E - mu_W) + Hy_fundo * (mu_N - mu_S)

                if denom > 1e-12:
                    psi_new = (mu_E * psi_tilde[y, x + 1] + mu_W * psi_tilde[y, x - 1] +
                               mu_N * psi_tilde[yp, x] + mu_S * psi_tilde[ym, x] - rhs) / denom

                    psi_tilde[y, x] = (1.0 - sor_omega) * psi_tilde[y, x] + sor_omega * psi_new

        # Condição de Neumann estrita no eixo X (a perturbação anula-se longe da interface)
        for y in prange(ny):
            psi_tilde[y, 0] = psi_tilde[y, 1]
            psi_tilde[y, nx - 1] = psi_tilde[y, nx - 2]

    return psi_tilde