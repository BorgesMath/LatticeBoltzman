# cahn_hilliard/cahn_hilliard.py
import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def cahn_hilliard_substep(phi_in, phi_out, mu, u_x, u_y, beta, kappa, dt_ch, m_mobility):
    """
    Evolução da interface via equação de Cahn-Hilliard.
    Double buffering: lê de phi_in e escreve em phi_out.
    """
    ny, nx = phi_in.shape

    # Cálculo do Potencial Químico (Armazenado no buffer mu)
    for y in prange(1, ny - 1):
        for x in range(1, nx - 1):
            # Laplaciano de phi (stencil de 5 pontos)
            lap_phi = (phi_in[y, x + 1] + phi_in[y, x - 1] +
                       phi_in[y + 1, x] + phi_in[y - 1, x] - 4.0 * phi_in[y, x])

            # mu = df/dphi - kappa*lap(phi)
            mu[y, x] = 4.0 * beta * phi_in[y, x] * (phi_in[y, x] ** 2 - 1.0) - kappa * lap_phi

    # Evolução Temporal (Advecção-Difusão)
    for y in prange(1, ny - 1):
        for x in range(1, nx - 1):
            # Laplaciano do Potencial Químico
            lap_mu = (mu[y, x + 1] + mu[y, x - 1] +
                      mu[y + 1, x] + mu[y - 1, x] - 4.0 * mu[y, x])

            # Termo Advectivo (Upwind ou Diferença Central - aqui Central para conservação)
            dphi_dx = 0.5 * (phi_in[y, x + 1] - phi_in[y, x - 1])
            dphi_dy = 0.5 * (phi_in[y + 1, x] - phi_in[y - 1, x])

            advec = u_x[y, x] * dphi_dx + u_y[y, x] * dphi_dy

            # Atualização em phi_out
            phi_out[y, x] = phi_in[y, x] + dt_ch * (m_mobility * lap_mu - advec)

    # Condições de Contorno de Dirichlet e Neumann (Inlet/Outlet)
    for y in prange(ny):
        phi_out[y, 0] = 1.0  # Fluido invasor puro na entrada
        phi_out[y, nx - 1] = phi_out[y, nx - 2]  # Gradiente zero na saída