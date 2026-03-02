# main.py
import numpy as np
from tqdm import tqdm

from config.config import MAX_ITER, SNAPSHOT_STEPS, CHI_MAX, CH_SUBSTEPS
from initialization.initialization import initialize_fields
from poisson.poisson import solve_poisson_magnetic
from cahn_hilliard.cahn_hilliard import cahn_hilliard_substep
from lbm.lbm import lbm_step
from post_process import post_process


def run_simulation(mode_m=4, amplitude=5.0):
    """
    Orquestra a simulação da Instabilidade de Saffman-Taylor em meio poroso.
    O módulo analítico (LSA) foi desacoplado desta versão.
    """
    base_dir = post_process.setup_output_dir(mode_m)

    # Alocação de Tensores e Definição do Problema de Valor Inicial
    f, phi, psi, rho, u_x, u_y, K_field = initialize_fields(mode_m, amplitude)

    checkpoints = np.linspace(0, MAX_ITER - 1, SNAPSHOT_STEPS, dtype=int)
    mass_history = np.zeros(MAX_ITER, dtype=np.float64)
    curv_history = np.zeros(MAX_ITER, dtype=np.float64)
    time_steps = np.arange(MAX_ITER)

    # Integração Temporal Numérica
    for t in tqdm(range(MAX_ITER), desc=f"Integração Numérica (Modo {mode_m})"):

        # Solver Elíptico
        chi_field = np.clip((phi + 1.0) * 0.5, 0.0, 1.0) * CHI_MAX
        psi = solve_poisson_magnetic(psi, chi_field)

        # Dinâmica Interfacial
        for _ in range(CH_SUBSTEPS):
            phi = cahn_hilliard_substep(phi, u_x, u_y)

        # Cinética de Boltzmann acoplada à Força de Darcy-Brinkman
        f, rho, u_x, u_y = lbm_step(f, phi, psi, rho, u_x, u_y, chi_field, K_field)

        # Diagnósticos Invariantes
        mass_history[t] = np.sum(rho)
        curv_history[t] = post_process.compute_interface_curvature(phi)

        # Exportação de Campos Espaciais
        if t in checkpoints:
            post_process.export_fields(phi, psi, rho, u_x, u_y, mode_m, t, base_dir)

    # Exportação Final
    post_process.export_time_series(mass_history, curv_history, time_steps, mode_m, base_dir)

    post_process.export_tip_position(phi, mode_m, base_dir)

    print(f"\nIntegração concluída. Diagnósticos em: {base_dir}")


if __name__ == "__main__":
    run_simulation(mode_m=4, amplitude=1.0)