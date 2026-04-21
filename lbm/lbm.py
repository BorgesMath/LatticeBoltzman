import numpy as np
from numba import njit, prange

# Constantes D2Q9 (Otimizadas para registradores)
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36], dtype=np.float64)
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)


@njit(parallel=True, cache=True)
def lbm_step(f_in, f_out, phi, psi, rho, u_x, u_y, chi, K_field, Fx, Fy,
             tau_in, tau_out, u_inlet, beta, kappa):
    ny, nx, _ = f_in.shape

    # 1. Cálculo de Forças Externas (Capilar + Magnética)
    for y in prange(1, ny - 1):
        for x in range(1, nx - 1):
            # Força Capilar (Korteweg)
            dx_phi = 0.5 * (phi[y, x + 1] - phi[y, x - 1])
            dy_phi = 0.5 * (phi[y + 1, x] - phi[y - 1, x])
            lap_phi = (phi[y, x + 1] + phi[y, x - 1] + phi[y + 1, x] + phi[y - 1, x] - 4.0 * phi[y, x])
            mu_c = 4.0 * beta * phi[y, x] * (phi[y, x] ** 2 - 1.0) - kappa * lap_phi

            Fx[y, x] = mu_c * dx_phi
            Fy[y, x] = mu_c * dy_phi

            # Força Magnética (Kelvin Force rigorosa via Hessiano)
            hx = -0.5 * (psi[y, x + 1] - psi[y, x - 1])
            hy = -0.5 * (psi[y + 1, x] - psi[y - 1, x])

            d2psi_dx2 = psi[y, x + 1] - 2.0 * psi[y, x] + psi[y, x - 1]
            d2psi_dy2 = psi[y + 1, x] - 2.0 * psi[y, x] + psi[y - 1, x]
            d2psi_dxy = 0.25 * (psi[y + 1, x + 1] - psi[y + 1, x - 1] - psi[y - 1, x + 1] + psi[y - 1, x - 1])

            Fx[y, x] += chi[y, x] * (hx * (-d2psi_dx2) + hy * (-d2psi_dxy))
            Fy[y, x] += chi[y, x] * (hx * (-d2psi_dxy) + hy * (-d2psi_dy2))

    # 2. Colisão e Streaming (com Periodicidade em Y)
    nu_in = (tau_in - 0.5) / 3.0
    nu_out = (tau_out - 0.5) / 3.0

    for y in prange(ny):
        for x in prange(nx):
            S_inv = (phi[y, x] + 1.0) * 0.5
            S_res = 1.0 - S_inv

            tau = tau_out + (tau_in - tau_out) * S_inv
            omega = 1.0 / tau

            # Modelo de Brinkman para Meio Poroso
            kr_inv = max(S_inv ** 2, 1e-6)
            kr_res = max(S_res ** 2, 1e-6)
            lambda_total = (kr_inv / nu_in) + (kr_res / nu_out)
            sigma_drag = 1.0 / (K_field[y, x] * lambda_total)

            # Cálculo dos Momentos Locais
            rho_l = 0.0
            ux_l = 0.0
            uy_l = 0.0
            for i in range(9):
                rho_l += f_in[y, x, i]
                ux_l += f_in[y, x, i] * CX[i]
                uy_l += f_in[y, x, i] * CY[i]

            # Velocidade Estrela (Equilíbrio sem Drag)
            ux_star = (ux_l + 0.5 * Fx[y, x]) / rho_l
            uy_star = (uy_l + 0.5 * Fy[y, x]) / rho_l

            # Velocidade Física (Corrigida por Brinkman/Darcy)
            ux_phys = ux_star / (1.0 + 0.5 * sigma_drag)
            uy_phys = uy_star / (1.0 + 0.5 * sigma_drag)

            rho[y, x], u_x[y, x], u_y[y, x] = rho_l, ux_phys, uy_phys
            u_sq = ux_phys ** 2 + uy_phys ** 2

            # Força Total (Externa + Drag)
            Fx_total = Fx[y, x] - (sigma_drag * rho_l * ux_phys)
            Fy_total = Fy[y, x] - (sigma_drag * rho_l * uy_phys)

            for i in range(9):
                cu = CX[i] * ux_phys + CY[i] * uy_phys
                feq = W[i] * rho_l * (1.0 + 3.0 * cu + 4.5 * cu ** 2 - 1.5 * u_sq)

                # Termo de Força de Guo
                term1 = (CX[i] - ux_phys) * Fx_total + (CY[i] - uy_phys) * Fy_total
                term2 = cu * (CX[i] * Fx_total + CY[i] * Fy_total)
                Si = W[i] * (1.0 - 0.5 * omega) * (3.0 * term1 + 9.0 * term2)

                f_val = f_in[y, x, i] * (1.0 - omega) + omega * feq + Si

                # Streaming Periódico no Eixo Y, Aberto no Eixo X
                be_y = (y + CY[i]) % ny
                be_x = x + CX[i]

                if 0 <= be_x < nx:
                    f_out[be_y, be_x, i] = f_val

    # 3. Condições de Fronteira
    for y in prange(ny):
        # Outlet Neumann (Gradiente Zero)
        for i in range(9):
            f_out[y, nx - 1, i] = f_out[y, nx - 2, i]

        # Inlet (Zou-He Velocity Boundary Condition para u_x = u_inlet, u_y = 0)
        # Calcula a densidade instantânea induzida pela contrapressão do meio poroso
        rho_in = (1.0 / (1.0 - u_inlet)) * (
                f_out[y, 0, 0] + f_out[y, 0, 2] + f_out[y, 0, 4] +
                2.0 * (f_out[y, 0, 3] + f_out[y, 0, 6] + f_out[y, 0, 7])
        )

        # Reconstrói as populações entrantes (Leste, Nordeste, Sudeste) conservando o momento
        f_out[y, 0, 1] = f_out[y, 0, 3] + (2.0 / 3.0) * rho_in * u_inlet
        f_out[y, 0, 5] = f_out[y, 0, 7] - 0.5 * (f_out[y, 0, 2] - f_out[y, 0, 4]) + (1.0 / 6.0) * rho_in * u_inlet
        f_out[y, 0, 8] = f_out[y, 0, 6] + 0.5 * (f_out[y, 0, 2] - f_out[y, 0, 4]) + (1.0 / 6.0) * rho_in * u_inlet

    return f_out, rho, u_x, u_y