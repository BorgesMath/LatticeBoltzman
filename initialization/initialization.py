# initialization/initialization.py
import numpy as np
from numba import njit, prange

# Tensores D2Q9
W_LBM = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36], dtype=np.float64)
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32)


@njit(parallel=True, cache=True)
def _init_kernel(phi, f, psi, rho, u_x, ny, nx, mode_m, amplitude, interface_width, x_center, Hx, Hy, u_inlet,
                 rho_base):
    u_sq = u_inlet ** 2

    for y in prange(ny):
        dist = x_center + amplitude * np.cos(2.0 * np.pi * mode_m * y / ny)

        for x in range(nx):
            phi[y, x] = -np.tanh((x - dist) / interface_width)
            psi[y, x] = Hx * (nx - x) + Hy * (ny - y)
            u_x[y, x] = u_inlet

            # Injeção da pressão analítica pré-calculada
            rho_local = rho_base[x]
            rho[y, x] = rho_local

            for i in range(9):
                cu = CX[i] * u_inlet
                f[y, x, i] = W_LBM[i] * rho_local * (1.0 + 3.0 * cu + 4.5 * (cu ** 2) - 1.5 * u_sq)


def initialize_fields(params):
    ny, nx = params["NY"], params["NX"]
    u_inlet = params["U_INLET"]
    x_center = 80.0

    # =========================================================================
    # CÁLCULO ANALÍTICO DO FLUXO BASE DE DARCY (Evita decaimento da velocidade)
    # =========================================================================
    rho_base = np.ones(nx, dtype=np.float64)
    nu_in = (params["TAU_IN"] - 0.5) / 3.0
    nu_out = (params["TAU_OUT"] - 0.5) / 3.0

    dpdx_in = 3.0 * (nu_in / params["K_0"]) * u_inlet
    dpdx_out = 3.0 * (nu_out / params["K_0"]) * u_inlet

    # Integração cumulativa de Trás para Frente (Outlet ancorado em rho=1.0)
    for x in range(nx - 2, -1, -1):
        if x >= x_center:
            rho_base[x] = rho_base[x + 1] + dpdx_out
        else:
            rho_base[x] = rho_base[x + 1] + dpdx_in
    # =========================================================================

    f_a = np.empty((ny, nx, 9), dtype=np.float64)
    f_b = np.empty((ny, nx, 9), dtype=np.float64)
    phi_a = np.empty((ny, nx), dtype=np.float64)
    phi_b = np.empty((ny, nx), dtype=np.float64)

    psi = np.empty((ny, nx), dtype=np.float64)
    rho = np.empty((ny, nx), dtype=np.float64)
    u_x = np.empty((ny, nx), dtype=np.float64)
    u_y = np.zeros((ny, nx), dtype=np.float64)
    K_field = np.ones((ny, nx), dtype=np.float64) * params["K_0"]

    Fx = np.zeros((ny, nx), dtype=np.float64)
    Fy = np.zeros((ny, nx), dtype=np.float64)
    mu_buffer = np.zeros((ny, nx), dtype=np.float64)

    angle_rad = np.radians(params["H_ANGLE"])
    Hx = params["H0"] * np.cos(angle_rad)
    Hy = params["H0"] * np.sin(angle_rad)

    # Passamos o rho e o rho_base para o kernel
    _init_kernel(phi_a, f_a, psi, rho, u_x, ny, nx, params["mode_m"],
                 params["amplitude"], params["INTERFACE_WIDTH"], x_center, Hx, Hy, u_inlet, rho_base)

    f_b[:] = f_a[:]
    phi_b[:] = phi_a[:]

    return (f_a, f_b), (phi_a, phi_b), psi, rho, u_x, u_y, K_field, (Fx, Fy, mu_buffer)