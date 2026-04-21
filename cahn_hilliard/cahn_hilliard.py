import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def cahn_hilliard_substep(phi_in, phi_out, mu, u_x, u_y, beta, kappa, dt_ch, m_mobility):
    ny, nx = phi_in.shape

    # 1. Sincronização de Contorno em phi_in (Garante continuidade do stencil)
    for y in prange(ny):
        phi_in[y, 0] = 1.0  # Inlet (Fluido invasor puro)
        phi_in[y, nx - 1] = phi_in[y, nx - 2]  # Outlet (Neumann nulo)
    for x in prange(nx):
        phi_in[0, x] = phi_in[1, x]  # Parede Inferior
        phi_in[ny - 1, x] = phi_in[ny - 2, x]  # Parede Superior

    # 2. Cálculo do Potencial Químico no bulk
    for y in prange(1, ny - 1):
        for x in range(1, nx - 1):
            lap_phi = (phi_in[y, x + 1] + phi_in[y, x - 1] +
                       phi_in[y + 1, x] + phi_in[y - 1, x] - 4.0 * phi_in[y, x])
            mu[y, x] = 4.0 * beta * phi_in[y, x] * (phi_in[y, x] ** 2 - 1.0) - kappa * lap_phi

    # 3. Sincronização de Contorno em mu (CRÍTICO para evitar fluxo infinito nas bordas)
    for y in prange(ny):
        mu[y, 0] = mu[y, 1]
        mu[y, nx - 1] = mu[y, nx - 2]
    for x in prange(nx):
        mu[0, x] = mu[1, x]
        mu[ny - 1, x] = mu[ny - 2, x]

    # 4. Evolução Temporal (Advecção-Difusão)
    for y in prange(1, ny - 1):
        for x in range(1, nx - 1):
            lap_mu = (mu[y, x + 1] + mu[y, x - 1] +
                      mu[y + 1, x] + mu[y - 1, x] - 4.0 * mu[y, x])

            dphi_dx = 0.5 * (phi_in[y, x + 1] - phi_in[y, x - 1])
            dphi_dy = 0.5 * (phi_in[y + 1, x] - phi_in[y - 1, x])

            advec = u_x[y, x] * dphi_dx + u_y[y, x] * dphi_dy

            # Atualização
            novo_phi = phi_in[y, x] + dt_ch * (m_mobility * lap_mu - advec)

            # Clipagem de estabilidade termodinâmica restrita a [-1, 1]
            if novo_phi > 1.0:
                novo_phi = 1.0
            elif novo_phi < -1.0:
                novo_phi = -1.0

            phi_out[y, x] = novo_phi

    # 5. Reaplicação do Contorno na saída do buffer
    for y in prange(ny):
        phi_out[y, 0] = 1.0
        phi_out[y, nx - 1] = phi_out[y, nx - 2]
    for x in prange(nx):
        phi_out[0, x] = phi_out[1, x]
        phi_out[ny - 1, x] = phi_out[ny - 2, x]