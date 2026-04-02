# Modificações no main.py
import math
from numba import cuda
import cupy as cp



# Parâmetros de topologia da GPU
threads_per_block = (16, 16)
blocks_per_grid_y = math.ceil(NY / threads_per_block[0])
blocks_per_grid_x = math.ceil(NX / threads_per_block[1])
blocks_per_grid = (blocks_per_grid_y, blocks_per_grid_x)


def run_simulation_gpu(mode_m=32, amplitude=2.0):
    # Inicialização na CPU
    f_cpu, phi_cpu, psi_cpu, rho_cpu, u_x_cpu, u_y_cpu, K_field_cpu = initialize_fields(mode_m, amplitude)

    # Transferência Host-to-Device (VRAM)
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

    for t in tqdm(range(MAX_ITER)):
        # Resolva a magnetostática (Se Poisson, substitua o solver do scipy/numpy por uma rotina CuPy)
        # psi_d = solve_poisson_magnetic_gpu(...)

        # Cahn-Hilliard Substeps
        for _ in range(CH_SUBSTEPS):
            calc_mu_kernel[blocks_per_grid, threads_per_block](phi_d, mu_d, NY, NX, BETA, KAPPA)
            cuda.synchronize()  # Barreira explícita para garantir o cálculo de vizinhança

            ch_advance_kernel[blocks_per_grid, threads_per_block](phi_d, phi_next_d, mu_d, u_x_d, u_y_d, NY, NX, DT_CH,
                                                                  M_MOBILITY)
            cuda.synchronize()

            # Swap de ponteiros (Evita cópia de memória)
            phi_d, phi_next_d = phi_next_d, phi_d

        # LBM Step
        lbm_forces_kernel[blocks_per_grid, threads_per_block](phi_d, psi_d, chi_field_d, Fx_d, Fy_d, NY, NX, BETA,
                                                              KAPPA)
        cuda.synchronize()

        lbm_collision_streaming_kernel[blocks_per_grid, threads_per_block](f_d, f_new_d, phi_d, rho_d, u_x_d, u_y_d,
                                                                           Fx_d, Fy_d, K_field_d, NY, NX, TAU_IN,
                                                                           TAU_OUT)
        cuda.synchronize()

        # Swap dos tensores de densidade de probabilidade
        f_d, f_new_d = f_new_d, f_d

        if t in checkpoints:
            # Transferência Device-to-Host (Apenas em checkpoints de salvamento)
            phi_cpu = phi_d.copy_to_host()
            rho_cpu = rho_d.copy_to_host()
            u_x_cpu = u_x_d.copy_to_host()
            u_y_cpu = u_y_d.copy_to_host()
            post_process.export_fields(phi_cpu, psi_cpu, rho_cpu, u_x_cpu, u_y_cpu, mode_m, t, base_dir)