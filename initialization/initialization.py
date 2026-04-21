# initialization/initialization.py
import numpy as np
from numba import njit, prange

# Tensores D2Q9
W_LBM = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36], dtype=np.float64)
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32)


@njit(parallel=True, cache=True)
def _init_kernel(phi, f, psi, u_x, ny, nx, mode_m, amplitude, interface_width, x_center, Hx, Hy, u_inlet):
    u_sq = u_inlet ** 2

    for y in prange(ny):
        # Condição da interface perturbada analiticamente
        dist = x_center + amplitude * np.cos(2.0 * np.pi * mode_m * y / ny)

        for x in range(nx):
            phi[y, x] = -np.tanh((x - dist) / (interface_width / 2.0))
            psi[y, x] = Hx * (nx - x) + Hy * (ny - y)
            u_x[y, x] = u_inlet

            # Inicialização termodinâmica rigorosa: f = f_eq(rho=1.0, u=u_inlet)
            for i in range(9):
                cu = CX[i] * u_inlet  # u_y é assumido 0 neste instante
                f[y, x, i] = W_LBM[i] * 1.0 * (1.0 + 3.0 * cu + 4.5 * (cu ** 2) - 1.5 * u_sq)


def initialize_fields(params):
    ny, nx = params["NY"], params["NX"]
    u_inlet = params["U_INLET"]

    # Tensores de Evolução (Double Buffering)
    f_a = np.empty((ny, nx, 9), dtype=np.float64)
    f_b = np.empty((ny, nx, 9), dtype=np.float64)
    phi_a = np.empty((ny, nx), dtype=np.float64)
    phi_b = np.empty((ny, nx), dtype=np.float64)

    # Campos de Estado e Auxiliares
    psi = np.empty((ny, nx), dtype=np.float64)
    rho = np.ones((ny, nx), dtype=np.float64)
    u_x = np.empty((ny, nx), dtype=np.float64)
    u_y = np.zeros((ny, nx), dtype=np.float64)  # Dinâmica inicial predominantemente horizontal
    K_field = np.ones((ny, nx), dtype=np.float64) * params["K_0"]

    # Buffers Dinâmicos do Kernel
    Fx = np.zeros((ny, nx), dtype=np.float64)
    Fy = np.zeros((ny, nx), dtype=np.float64)
    mu_buffer = np.zeros((ny, nx), dtype=np.float64)

    # Derivação do campo magnético externo
    angle_rad = np.radians(params["H_ANGLE"])
    Hx = params["H0"] * np.cos(angle_rad)
    Hy = params["H0"] * np.sin(angle_rad)

    _init_kernel(phi_a, f_a, psi, u_x, ny, nx, params["mode_m"],
                 params["amplitude"], params["INTERFACE_WIDTH"], 80.0, Hx, Hy, u_inlet)

    # Sincronização do estado inicial 'b' a partir de 'a'
    f_b[:] = f_a[:]
    phi_b[:] = phi_a[:]

    return (f_a, f_b), (phi_a, phi_b), psi, rho, u_x, u_y, K_field, (Fx, Fy, mu_buffer)