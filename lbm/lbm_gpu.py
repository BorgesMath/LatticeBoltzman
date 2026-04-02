# lbm/lbm_gpu.py
from numba import cuda


# Tensores constantes na memória da GPU
@cuda.jit(device=True)
def get_cx(i):
    cx = (0, 1, 0, -1, 0, 1, -1, -1, 1)
    return cx[i]


@cuda.jit(device=True)
def get_cy(i):
    cy = (0, 0, 1, 0, -1, 1, 1, -1, -1)
    return cy[i]


@cuda.jit(device=True)
def get_w(i):
    w = (4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36)
    return w[i]


@cuda.jit
def lbm_forces_kernel(phi, psi, chi_field, Fx, Fy, ny, nx, BETA, KAPPA):
    y, x = cuda.grid(2)
    if 0 < y < ny - 1 and 0 < x < nx - 1:
        # Tensão de Korteweg
        dx_phi = 0.5 * (phi[y, x + 1] - phi[y, x - 1])
        dy_phi = 0.5 * (phi[y + 1, x] - phi[y - 1, x])
        lap_phi = phi[y, x + 1] + phi[y, x - 1] + phi[y + 1, x] + phi[y - 1, x] - 4.0 * phi[y, x]
        mu_c = 4.0 * BETA * phi[y, x] * (phi[y, x] ** 2 - 1.0) - KAPPA * lap_phi

        fx_loc = mu_c * dx_phi
        fy_loc = mu_c * dy_phi

        # Força Magnética de Kelvin
        hx = -0.5 * (psi[y, x + 1] - psi[y, x - 1])
        hy = -0.5 * (psi[y + 1, x] - psi[y - 1, x])
        d2psi_dx2 = psi[y, x + 1] - 2 * psi[y, x] + psi[y, x - 1]
        d2psi_dy2 = psi[y + 1, x] - 2 * psi[y, x] + psi[y - 1, x]
        d2psi_dxy = 0.25 * (psi[y + 1, x + 1] - psi[y + 1, x - 1] - psi[y - 1, x + 1] + psi[y - 1, x - 1])

        fx_loc += chi_field[y, x] * (hx * (-d2psi_dx2) + hy * (-d2psi_dxy))
        fy_loc += chi_field[y, x] * (hx * (-d2psi_dxy) + hy * (-d2psi_dy2))

        Fx[y, x] = fx_loc
        Fy[y, x] = fy_loc


@cuda.jit
def lbm_collision_streaming_kernel(f, f_new, phi, rho, u_x, u_y, Fx, Fy, K_field, ny, nx, TAU_IN, TAU_OUT):
    y, x = cuda.grid(2)
    if y < ny and x < nx:
        S_inv = max(0.0, min(1.0, (phi[y, x] + 1.0) * 0.5))
        tau = TAU_OUT + (TAU_IN - TAU_OUT) * S_inv
        omega = 1.0 / tau
        nu_local = (tau - 0.5) / 3.0
        sigma_drag = nu_local / K_field[y, x]

        rho_l, ux_l, uy_l = 0.0, 0.0, 0.0
        for i in range(9):
            rho_l += f[y, x, i]
            ux_l += f[y, x, i] * get_cx(i)
            uy_l += f[y, x, i] * get_cy(i)

        ux_star = (ux_l + 0.5 * Fx[y, x]) / rho_l
        uy_star = (uy_l + 0.5 * Fy[y, x]) / rho_l

        ux_phys = ux_star / (1.0 + 0.5 * sigma_drag)
        uy_phys = uy_star / (1.0 + 0.5 * sigma_drag)

        rho[y, x], u_x[y, x], u_y[y, x] = rho_l, ux_phys, uy_phys
        usq = ux_phys ** 2 + uy_phys ** 2

        Fx_total = Fx[y, x] - (sigma_drag * rho_l * ux_phys)
        Fy_total = Fy[y, x] - (sigma_drag * rho_l * uy_phys)

        for i in range(9):
            cx_i, cy_i, w_i = get_cx(i), get_cy(i), get_w(i)
            cu = cx_i * ux_phys + cy_i * uy_phys
            feq = w_i * rho_l * (1.0 + 3.0 * cu + 4.5 * cu ** 2 - 1.5 * usq)

            term1 = (cx_i - ux_phys) * Fx_total + (cy_i - uy_phys) * Fy_total
            term2 = cu * (cx_i * Fx_total + cy_i * Fy_total)
            Si = w_i * (1.0 - 0.5 * omega) * (3.0 * term1 + 9.0 * term2)

            f_val = f[y, x, i] * (1.0 - omega) + omega * feq + Si

            be_x = x + cx_i
            be_y = y + cy_i

            if 0 <= be_y < ny:
                if 0 <= be_x < nx:
                    f_new[be_y, be_x, i] = f_val
            else:
        # Bounce-back simples mapeando o índice oposto (OPP iterativo pode ser implementado em função device)
        # ... Lógica de mapeamento reverso ...