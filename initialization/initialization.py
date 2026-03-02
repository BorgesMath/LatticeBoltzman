# initialization.py
import numpy as np
from config.config import NY, NX, W_LBM, H0, INTERFACE_WIDTH, K_0 # Correção aqui

def initialize_fields(mode_m, amplitude):
    """
    Aloca os tensores e define o Problema de Valor Inicial (PVI).
    """
    f = np.zeros((NY, NX, 9), dtype=np.float64)
    phi = np.zeros((NY, NX), dtype=np.float64)
    psi = np.zeros((NY, NX), dtype=np.float64)
    rho = np.ones((NY, NX), dtype=np.float64)
    u_x = np.zeros((NY, NX), dtype=np.float64)
    u_y = np.zeros((NY, NX), dtype=np.float64)

    # Matriz de Permeabilidade Absoluta alocada a partir do config.py
    K_field = np.ones((NY, NX), dtype=np.float64) * K_0

    x_center = 80.0
    for y in range(NY):
        dist = x_center + amplitude * np.cos(2.0 * np.pi * mode_m * y / NY)
        for x in range(NX):
            phi[y, x] = -np.tanh((x - dist) / (INTERFACE_WIDTH / 2.0))
            for i in range(9):
                f[y, x, i] = W_LBM[i] * rho[y, x]
            psi[y, x] = H0 * (NX - x)

    return f, phi, psi, rho, u_x, u_y, K_field