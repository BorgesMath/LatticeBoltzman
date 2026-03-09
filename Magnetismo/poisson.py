# Magnetismo/poisson.py
import numpy as np
from numba import njit
from config.config import SOR_OMEGA, H0, H_ANGLE


@njit
def solve_poisson_magnetic(psi, chi_field):
    """
    Resolve a equação de Poisson para o potencial magnético escalar
    via Sucessiva Sobre-Relaxação (SOR).

    ATUALIZAÇÃO:
    - Agora suporta campo magnético angulado (H_ANGLE).
    - Aplica condições de contorno de Neumann condizentes com a componente vertical (Hy).
    """
    ny, nx = psi.shape

    # Decomposição do vetor campo magnético externo desejado
    # H_ANGLE em graus (0 = Horizontal/X, 90 = Vertical/Y)
    angle_rad = np.radians(H_ANGLE)

    # Componente vertical do campo (Hy)
    # Como H = -grad(psi), então Hy = -dPsi/dy
    # Logo, a variação de Psi em Y deve ser dPsi = -Hy * dy
    Hy_target = H0 * np.sin(angle_rad)

    # Fator de correção para as bordas (dPsi_y)
    # Assumindo dy = 1 (unidades de rede)
    dPsi_y = -Hy_target

    for _ in range(15):  # Iterações SOR internas por passo de tempo LBM
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                # Interpolação da permeabilidade magnética nas faces do volume de controle
                mu_E = 1.0 + 0.5 * (chi_field[y, x] + chi_field[y, x + 1])
                mu_W = 1.0 + 0.5 * (chi_field[y, x] + chi_field[y, x - 1])
                mu_N = 1.0 + 0.5 * (chi_field[y, x] + chi_field[y + 1, x])
                mu_S = 1.0 + 0.5 * (chi_field[y, x] + chi_field[y - 1, x])

                denom = mu_E + mu_W + mu_N + mu_S

                if denom > 1e-12:
                    psi_new = (mu_E * psi[y, x + 1] + mu_W * psi[y, x - 1] +
                               mu_N * psi[y + 1, x] + mu_S * psi[y - 1, x]) / denom

                    # Atualização SOR
                    psi[y, x] = (1.0 - SOR_OMEGA) * psi[y, x] + SOR_OMEGA * psi_new

        # =========================================================
        # CONDIÇÕES DE CONTORNO (ATUALIZADAS PARA ÂNGULO)
        # =========================================================

        # 1. Paredes Transversais (Topo e Fundo) - Condição de Neumann
        # Permite fluxo magnético vertical se H_ANGLE != 0
        # H_y = -(psi[y+1] - psi[y]) => psi[y+1] = psi[y] - H_y

        # Fundo (y=0): psi[0] deve ser maior que psi[1] se o campo aponta para cima
        # Aprox: (psi[1] - psi[0])/1 = -Hy  => psi[0] = psi[1] + Hy
        psi[0, :] = psi[1, :] + Hy_target

        # Topo (y=ny-1):
        # Aprox: (psi[end] - psi[end-1])/1 = -Hy => psi[end] = psi[end-1] - Hy
        psi[ny - 1, :] = psi[ny - 2, :] - Hy_target

        # Nota: As bordas Esquerda (x=0) e Direita (x=nx-1) são Dirichlet (fixas).
        # Elas não são atualizadas aqui (o loop de x vai de 1 a nx-1),
        # mantendo o gradiente horizontal imposto na inicialização.

    return psi