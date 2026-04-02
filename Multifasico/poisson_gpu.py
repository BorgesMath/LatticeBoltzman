import math
import numpy as np
from numba import cuda

@cuda.jit
def poisson_sor_red_black_kernel(psi, chi_field, ny, nx, SOR_OMEGA, color):
    """
    Kernel CUDA para iterar a equação de Poisson usando Red-Black SOR.
    color = 0 (Vermelho) ou 1 (Preto).
    """
    y, x = cuda.grid(2)

    # Garante que a operação ocorra apenas nos nós internos
    if 0 < y < ny - 1 and 0 < x < nx - 1:
        # Condição de particionamento Red-Black
        if (y + x) % 2 == color:
            mu_E = 1.0 + 0.5 * (chi_field[y, x] + chi_field[y, x + 1])
            mu_W = 1.0 + 0.5 * (chi_field[y, x] + chi_field[y, x - 1])
            mu_N = 1.0 + 0.5 * (chi_field[y, x] + chi_field[y + 1, x])
            mu_S = 1.0 + 0.5 * (chi_field[y, x] + chi_field[y - 1, x])

            denom = mu_E + mu_W + mu_N + mu_S

            if denom > 1e-12:
                psi_new = (mu_E * psi[y, x + 1] + mu_W * psi[y, x - 1] +
                           mu_N * psi[y + 1, x] + mu_S * psi[y - 1, x]) / denom

                # Atualização no local (seguro devido ao Red-Black)
                psi[y, x] = (1.0 - SOR_OMEGA) * psi[y, x] + SOR_OMEGA * psi_new

@cuda.jit
def apply_neumann_boundaries_kernel(psi, ny, nx, Hy_target):
    """
    Kernel 1D para aplicar as condições de contorno transversais.
    """
    x = cuda.grid(1)
    if x < nx:
        # Fundo (y=0)
        psi[0, x] = psi[1, x] + Hy_target
        # Topo (y=ny-1)
        psi[ny - 1, x] = psi[ny - 2, x] - Hy_target

def solve_poisson_magnetic_gpu(psi_d, chi_field_d, ny, nx, SOR_OMEGA, H0, H_ANGLE):
    """
    Função controladora (Wrapper) executada na CPU para orquestrar os kernels na GPU.
    Recebe ponteiros de memória de dispositivo (Device Arrays) do CuPy ou Numba.
    """
    # Cálculo trigonométrico feito na CPU uma única vez
    angle_rad = math.radians(H_ANGLE)
    Hy_target = H0 * math.sin(angle_rad)

    # Configuração da topologia de blocos 2D (Para a malha principal)
    threads_per_block_2d = (16, 16)
    blocks_per_grid_y = math.ceil(ny / threads_per_block_2d[0])
    blocks_per_grid_x = math.ceil(nx / threads_per_block_2d[1])
    blocks_grid_2d = (blocks_per_grid_y, blocks_per_grid_x)

    # Configuração da topologia de blocos 1D (Para os contornos)
    threads_per_block_1d = 256
    blocks_grid_1d = math.ceil(nx / threads_per_block_1d)

    for _ in range(15):
        # 1. Atualiza células Vermelhas
        poisson_sor_red_black_kernel[blocks_grid_2d, threads_per_block_2d](
            psi_d, chi_field_d, ny, nx, SOR_OMEGA, 0
        )
        cuda.synchronize() # Barreira de sincronização rigorosa

        # 2. Atualiza células Pretas
        poisson_sor_red_black_kernel[blocks_grid_2d, threads_per_block_2d](
            psi_d, chi_field_d, ny, nx, SOR_OMEGA, 1
        )
        cuda.synchronize()

        # 3. Impõe Condições de Contorno Magnéticas
        apply_neumann_boundaries_kernel[blocks_grid_1d, threads_per_block_1d](
            psi_d, ny, nx, Hy_target
        )
        cuda.synchronize()

    return psi_d