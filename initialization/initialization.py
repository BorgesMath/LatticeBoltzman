# initialization/initialization.py
import numpy as np
from config.config import NY, NX, W_LBM, H0, INTERFACE_WIDTH, K_0, H_ANGLE


def initialize_fields(mode_m, amplitude):
    """
    Aloca os tensores e define o Problema de Valor Inicial (PVI).

    ATUALIZAÇÃO:
    - Inicialização do campo magnético (psi) agora respeita o ângulo H_ANGLE.
    - Isso garante compatibilidade imediata com o solver de Poisson atualizado.
    """
    f = np.zeros((NY, NX, 9), dtype=np.float64)
    phi = np.zeros((NY, NX), dtype=np.float64)
    psi = np.zeros((NY, NX), dtype=np.float64)
    rho = np.ones((NY, NX), dtype=np.float64)
    u_x = np.zeros((NY, NX), dtype=np.float64)
    u_y = np.zeros((NY, NX), dtype=np.float64)

    # Matriz de Permeabilidade Absoluta alocada a partir do config.py
    K_field = np.ones((NY, NX), dtype=np.float64) * K_0

    # Decomposição do vetor campo magnético inicial
    # Se H_ANGLE = 0, Hx = H0, Hy = 0 (Recupera o comportamento original)
    angle_rad = np.radians(H_ANGLE)
    Hx = H0 * np.cos(angle_rad)
    Hy = H0 * np.sin(angle_rad)

    x_center = NX * (80.0 / 600.0) # Mantém a proporção da malha original

    for y in range(NY):
        # Perturbação da interface (Saffman-Taylor)
        dist = x_center + amplitude * np.cos(2.0 * np.pi * mode_m * y / NY)

        for x in range(NX):
            # 1. Campo de Fase (Phi) - Interface difusa (tanh)
            phi[y, x] = -np.tanh((x - dist) / (INTERFACE_WIDTH / 2.0))

            # 2. Distribuição inicial de partículas (f) - Equilíbrio em repouso
            for i in range(9):
                f[y, x, i] = W_LBM[i] * rho[y, x]

            # 3. Potencial Magnético Escalar (Psi)
            # Definido tal que H = -grad(Psi).
            # Queremos H = (Hx, Hy), então Psi = -(Hx*x + Hy*y) + C.
            # Usamos C = Hx*NX + Hy*NY para manter valores positivos (estética numérica),
            # resultando em: Psi = Hx*(NX - x) + Hy*(NY - y)
            psi[y, x] = Hx * (NX - x) + Hy * (NY - y)

    return f, phi, psi, rho, u_x, u_y, K_field