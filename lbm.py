import numpy as np
from numba import njit, prange
from config import (BETA, KAPPA, TAU_OUT, TAU_IN, W_LBM, CX, CY, OPP, U_INLET, PERIODIC_Y)


@njit(parallel=True)
def lbm_step(f, phi, psi, rho, u_x, u_y, chi_field, K_field):
    ny, nx, _ = f.shape
    Fx = np.zeros((ny, nx))
    Fy = np.zeros((ny, nx))

    # =========================================================
    # 1. CÁLCULO DO CAMPO DE FORÇAS MACROSCÓPICAS
    # =========================================================
    for y in prange(ny):
        # Roteamento topológico para cálculo de gradientes nas bordas do eixo Y
        y_up = (y + 1) % ny if PERIODIC_Y else min(y + 1, ny - 1)
        y_dn = (y - 1) % ny if PERIODIC_Y else max(y - 1, 0)

        for x in range(1, nx - 1):
            # Força de Korteweg (Capilar)
            dx_phi = 0.5 * (phi[y, x + 1] - phi[y, x - 1])
            dy_phi = 0.5 * (phi[y_up, x] - phi[y_dn, x])
            lap_phi = phi[y, x + 1] + phi[y, x - 1] + phi[y_up, x] + phi[y_dn, x] - 4.0 * phi[y, x]

            mu_c = 4.0 * BETA * phi[y, x] * (phi[y, x] ** 2 - 1.0) - KAPPA * lap_phi
            Fx[y, x] += mu_c * dx_phi
            Fy[y, x] += mu_c * dy_phi

            # Força de Kelvin (Magnética)
            hx = -0.5 * (psi[y, x + 1] - psi[y, x - 1])
            hy = -0.5 * (psi[y_up, x] - psi[y_dn, x])

            d2psi_dx2 = psi[y, x + 1] - 2 * psi[y, x] + psi[y, x - 1]
            d2psi_dy2 = psi[y_up, x] - 2 * psi[y, x] + psi[y_dn, x]
            d2psi_dxy = 0.25 * (psi[y_up, x + 1] - psi[y_up, x - 1] - psi[y_dn, x + 1] + psi[y_dn, x - 1])

            Fx[y, x] += chi_field[y, x] * (hx * (-d2psi_dx2) + hy * (-d2psi_dxy))
            Fy[y, x] += chi_field[y, x] * (hx * (-d2psi_dxy) + hy * (-d2psi_dy2))

    f_new = np.zeros_like(f)

    # Viscosidades cinemáticas assintóticas baseadas no relaxamento
    nu_in = (TAU_IN - 0.5) / 3.0
    nu_out = (TAU_OUT - 0.5) / 3.0

    # =========================================================
    # 2. OPERADOR DE COLISÃO BGK, ARRASTO DE DARCY E STREAMING
    # =========================================================
    for y in prange(ny):
        for x in prange(nx):
            # 2.1 Mapeamento Termodinâmico e Permeabilidade Relativa
            S_inv = max(0.0, min(1.0, (phi[y, x] + 1.0) * 0.5))
            S_res = 1.0 - S_inv

            tau = TAU_OUT + (TAU_IN - TAU_OUT) * S_inv
            omega = 1.0 / tau

            kr_inv = max(S_inv ** 2, 1e-6)
            kr_res = max(S_res ** 2, 1e-6)

            lambda_t = (kr_inv / nu_in) + (kr_res / nu_out)
            sigma_drag = 1.0 / (K_field[y, x] * lambda_t)

            # 2.2 Momentos Cinemáticos Provisórios
            rho_l = 0.0
            ux_l = 0.0
            uy_l = 0.0
            for i in range(9):
                rho_l += f[y, x, i]
                ux_l += f[y, x, i] * CX[i]
                uy_l += f[y, x, i] * CY[i]

            # 2.3 Resolução Implícita de Guo para Escoamento em Meios Porosos
            ux_star = (ux_l + 0.5 * Fx[y, x]) / rho_l
            uy_star = (uy_l + 0.5 * Fy[y, x]) / rho_l

            ux_phys = ux_star / (1.0 + 0.5 * sigma_drag)
            uy_phys = uy_star / (1.0 + 0.5 * sigma_drag)

            rho[y, x], u_x[y, x], u_y[y, x] = rho_l, ux_phys, uy_phys
            usq = ux_phys ** 2 + uy_phys ** 2

            Fx_total = Fx[y, x] - (sigma_drag * rho_l * ux_phys)
            Fy_total = Fy[y, x] - (sigma_drag * rho_l * uy_phys)

            # 2.4 Relaxação Estendida
            for i in range(9):
                cu = CX[i] * ux_phys + CY[i] * uy_phys
                feq = W_LBM[i] * rho_l * (1.0 + 3.0 * cu + 4.5 * cu ** 2 - 1.5 * usq)

                term1 = (CX[i] - ux_phys) * Fx_total + (CY[i] - uy_phys) * Fy_total
                term2 = cu * (CX[i] * Fx_total + CY[i] * Fy_total)
                Si = W_LBM[i] * (1.0 - 0.5 * omega) * (3.0 * term1 + 9.0 * term2)

                f_val = f[y, x, i] * (1.0 - omega) + omega * feq + Si

                # 2.5 Propagação com Topologia Seletiva (Eixo Y)
                be_x = x + int(CX[i])
                be_y = y + int(CY[i])

                if PERIODIC_Y:
                    be_y = be_y % ny
                    if 0 <= be_x < nx:
                        f_new[be_y, be_x, i] = f_val
                else:
                    if 0 <= be_y < ny:
                        if 0 <= be_x < nx:
                            f_new[be_y, be_x, i] = f_val
                    else:
                        f_new[y, x, OPP[i]] = f_val

    # =========================================================
    # 3. CONDIÇÕES DE CONTORNO MACROSCÓPICAS
    # =========================================================

    # 3.1 Outlet (Extrapolação de Ordem Zero / Neumann em X)
    for y in prange(ny):
        for i in range(9):
            f_new[y, nx - 1, i] = f_new[y, nx - 2, i]

    # 3.2 Inlet (Dirichlet de Velocidade Condicional)
    for y in prange(ny):
        if PERIODIC_Y:
            # Perfil Pistão (Plug Flow) para simetria livre
            u_in = U_INLET
        else:
            # Perfil Parabólico de Poiseuille para conformidade cinemática com Bounce-Back
            u_max = 1.5 * U_INLET
            y_norm = float(y) / float(ny - 1)
            u_in = 4.0 * u_max * y_norm * (1.0 - y_norm)

        for i in range(9):
            if CX[i] > 0:
                cu_in = CX[i] * u_in
                f_new[y, 0, i] = W_LBM[i] * 1.0 * (1.0 + 3.0 * cu_in + 4.5 * cu_in ** 2 - 1.5 * u_in ** 2)

    return f_new, rho, u_x, u_y