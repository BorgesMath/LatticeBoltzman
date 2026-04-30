import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def cahn_hilliard_substep(phi_in, phi_out, mu, u_x, u_y, beta, kappa, dt_ch, m_mobility, is_periodic):
    ny, nx = phi_in.shape

    # 1. Contornos em phi_in (apenas X; Y é sempre periódico)
    if not is_periodic:
        for y in prange(ny):
            phi_in[y, 0] = 1.0
            phi_in[y, nx - 1] = phi_in[y, nx - 2]

    # 2. Cálculo do Potencial Químico (Y sempre periódico)
    for y in prange(ny):
        yp = (y + 1) % ny
        ym = (y - 1 + ny) % ny
        for x in range(nx):
            if is_periodic:
                xp = (x + 1) % nx
                xm = (x - 1 + nx) % nx
            else:
                if x == 0 or x == nx - 1: continue
                xp, xm = x + 1, x - 1

            lap_phi = (phi_in[y, xp] + phi_in[y, xm] +
                       phi_in[yp, x] + phi_in[ym, x] - 4.0 * phi_in[y, x])
            mu[y, x] = 4.0 * beta * phi_in[y, x] * (phi_in[y, x] ** 2 - 1.0) - kappa * lap_phi

    # 3. Contornos em mu (apenas X)
    if not is_periodic:
        for y in prange(ny):
            mu[y, 0] = mu[y, 1]
            mu[y, nx - 1] = mu[y, nx - 2]

    # 4. Evolução Temporal (Advecção-Difusão, Y sempre periódico)
    for y in prange(ny):
        yp = (y + 1) % ny
        ym = (y - 1 + ny) % ny
        for x in range(nx):
            if is_periodic:
                xp = (x + 1) % nx
                xm = (x - 1 + nx) % nx
            else:
                if x == 0 or x == nx - 1: continue
                xp, xm = x + 1, x - 1

            lap_mu = (mu[y, xp] + mu[y, xm] + mu[yp, x] + mu[ym, x] - 4.0 * mu[y, x])
            dphi_dx = 0.5 * (phi_in[y, xp] - phi_in[y, xm])
            dphi_dy = 0.5 * (phi_in[yp, x] - phi_in[ym, x])

            advec = u_x[y, x] * dphi_dx + u_y[y, x] * dphi_dy
            novo_phi = phi_in[y, x] + dt_ch * (m_mobility * lap_mu - advec)

            if novo_phi > 1.0:
                novo_phi = 1.0
            elif novo_phi < -1.0:
                novo_phi = -1.0

            phi_out[y, x] = novo_phi

    # 5. Reaplicação do Contorno em phi_out (apenas X)
    if not is_periodic:
        for y in prange(ny):
            phi_out[y, 0] = 1.0
            phi_out[y, nx - 1] = phi_out[y, nx - 2]