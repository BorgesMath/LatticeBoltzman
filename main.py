import os
import sys
import ctypes

# 1. Identificação Topológica Determinística (CUDA 13.2)
cuda_path = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2'
bin_dir_x64 = os.path.join(cuda_path, 'bin', 'x64')
nvvm_bin_x64_dir = os.path.join(cuda_path, r'nvvm\bin\x64')

cudart_dll_path = os.path.join(bin_dir_x64, 'cudart64_13.dll')
nvvm_dll_path = os.path.join(nvvm_bin_x64_dir, 'nvvm64_40_0.dll')
libdevice_dir = os.path.join(cuda_path, r'nvvm\libdevice')

# 2. Mapeamento de Variáveis do Compilador Numba
os.environ['CUDA_HOME'] = cuda_path
os.environ['CUDA_PATH'] = cuda_path
os.environ['NUMBA_CUDA_NVVM'] = nvvm_dll_path
os.environ['NUMBA_CUDA_LIBDEVICE'] = libdevice_dir

# 3. Retenção de Escopo na Tabela de Resolução do PE Loader
cuda_dll_handles = []
if hasattr(os, 'add_dll_directory'):
    cuda_dll_handles.append(os.add_dll_directory(bin_dir_x64))
    cuda_dll_handles.append(os.add_dll_directory(nvvm_bin_x64_dir))

# 4. Trancamento de Memória (Memory Lock) das DLLs na RAM
old_cwd = os.getcwd()
try:
    os.chdir(bin_dir_x64)
    # Aloca as bibliotecas e SALVA os ponteiros para impedir o Garbage Collector de descarregá-las
    global_cudart_handle = ctypes.CDLL(cudart_dll_path, winmode=0)
    global_nvvm_handle = ctypes.CDLL(nvvm_dll_path, winmode=0)
except Exception as e:
    print(f"Erro fatal na pré-alocação das DLLs de C-Runtime: {e}", file=sys.stderr)
    sys.exit(1)
finally:
    os.chdir(old_cwd)

# ===============================================================
# INÍCIO DO ESCOPO DA APLICAÇÃO (INTEGRAÇÃO LBM)
# ===============================================================
import math
import numpy as np
import cupy as cp
from numba import cuda
from tqdm import tqdm

# ... (Mantenha as importações do seu projeto e o resto do main.py abaixo) ...

# ... (Mantenha o restante das suas importações e do código do main.py abaixo desta linha) ...

# Importação de constantes do domínio
from config.config import (NY, NX, MAX_ITER, SNAPSHOT_STEPS, CH_SUBSTEPS, BETA, KAPPA, DT_CH,
                           M_MOBILITY, TAU_IN, TAU_OUT, METODO_MAGNETISMO, SOR_OMEGA, H0, H_ANGLE, U_INLET)

# Importação de módulos
from initialization.initialization import initialize_fields
from Magnetismo.cahn_hilliard_gpu import calc_mu_kernel, ch_advance_kernel
from Multifasico.poisson_gpu import solve_poisson_magnetic_gpu
from lbm.lbm_gpu import lbm_forces_kernel, lbm_collision_streaming_kernel, lbm_boundaries_kernel

# Substitua pela sua função real de salvamento se aplicável
# from post_process import post_process

# Parâmetros de topologia da GPU (Malha 2D)
threads_per_block_2d = (16, 16)
blocks_per_grid_y = math.ceil(NY / threads_per_block_2d[0])
blocks_per_grid_x = math.ceil(NX / threads_per_block_2d[1])
blocks_per_grid_2d = (blocks_per_grid_y, blocks_per_grid_x)

# Parâmetros de topologia da GPU (Malha 1D para os Contornos Y)
threads_per_block_1d = 256
blocks_per_grid_1d_y = math.ceil(NY / threads_per_block_1d)


def run_simulation_gpu(mode_m=32, amplitude=2.0):
    base_dir = "resultados_gpu"
    os.makedirs(base_dir, exist_ok=True)

    # Define conjunto de passos para salvamento para mitigar o engarrafamento PCIe
    checkpoints = set(int(i) for i in np.linspace(0, MAX_ITER - 1, SNAPSHOT_STEPS))

    # Inicialização na CPU
    f_cpu, phi_cpu, psi_cpu, rho_cpu, u_x_cpu, u_y_cpu, K_field_cpu = initialize_fields(mode_m, amplitude)

    # Transferência Host-to-Device (Alocação na VRAM)
    f_d = cuda.to_device(f_cpu)
    f_new_d = cuda.device_array_like(f_d)
    phi_d = cuda.to_device(phi_cpu)
    phi_next_d = cuda.device_array_like(phi_d)
    psi_d = cuda.to_device(psi_cpu)
    rho_d = cuda.to_device(rho_cpu)
    u_x_d = cuda.to_device(u_x_cpu)
    u_y_d = cuda.to_device(u_y_cpu)
    K_field_d = cuda.to_device(K_field_cpu)

    mu_d = cuda.device_array((NY, NX), dtype=np.float64)
    Fx_d = cuda.device_array((NY, NX), dtype=np.float64)
    Fy_d = cuda.device_array((NY, NX), dtype=np.float64)
    chi_field_d = cuda.device_array((NY, NX), dtype=np.float64)

    for t in tqdm(range(MAX_ITER), desc="Iterações LBM na GPU"):

        # 1. Resolver Magnetostática (Red-Black SOR na VRAM)
        if METODO_MAGNETISMO == 'POISSON':
            solve_poisson_magnetic_gpu(psi_d, chi_field_d, NY, NX, SOR_OMEGA, H0, H_ANGLE)

        # 2. Dinâmica de Fase (Cahn-Hilliard)
        for _ in range(CH_SUBSTEPS):
            calc_mu_kernel[blocks_per_grid_2d, threads_per_block_2d](phi_d, mu_d, NY, NX, BETA, KAPPA)
            cuda.synchronize()

            ch_advance_kernel[blocks_per_grid_2d, threads_per_block_2d](phi_d, phi_next_d, mu_d, u_x_d, u_y_d, NY, NX,
                                                                        DT_CH, M_MOBILITY)
            cuda.synchronize()

            # Swap por apontador (Custo computacional O(1))
            phi_d, phi_next_d = phi_next_d, phi_d

        # 3. Forças Macroscópicas LBM
        lbm_forces_kernel[blocks_per_grid_2d, threads_per_block_2d](phi_d, psi_d, chi_field_d, Fx_d, Fy_d, NY, NX, BETA,
                                                                    KAPPA)
        cuda.synchronize()

        # 4. Operador de Colisão BGK + Streaming
        lbm_collision_streaming_kernel[blocks_per_grid_2d, threads_per_block_2d](f_d, f_new_d, phi_d, rho_d, u_x_d,
                                                                                 u_y_d, Fx_d, Fy_d, K_field_d, NY, NX,
                                                                                 TAU_IN, TAU_OUT)
        cuda.synchronize()

        # 5. Imposição de Condições de Contorno Macro LBM
        lbm_boundaries_kernel[blocks_per_grid_1d_y, threads_per_block_1d](f_new_d, rho_d, NY, NX, U_INLET)
        cuda.synchronize()

        f_d, f_new_d = f_new_d, f_d

        # 6. Checkpoint Device-to-Host
        if t in checkpoints:
            phi_cpu = phi_d.copy_to_host()
            rho_cpu = rho_d.copy_to_host()
            u_x_cpu = u_x_d.copy_to_host()
            u_y_cpu = u_y_d.copy_to_host()

            # Descomente e ajuste a chamada abaixo quando implementar o exportador de VTK/HDF5
            # post_process.export_fields(phi_cpu, psi_d.copy_to_host(), rho_cpu, u_x_cpu, u_y_cpu, mode_m, t, base_dir)


if __name__ == '__main__':
    run_simulation_gpu()