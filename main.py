# main.py
import json
import time
import numpy as np
from tqdm import tqdm

# Importação dos kernels otimizados e módulos de suporte
from initialization.initialization import initialize_fields
from Magnetismo.poisson import solve_poisson_magnetic
from cahn_hilliard.cahn_hilliard import cahn_hilliard_substep
from lbm.lbm import lbm_step
from post_process import post_process


def run_simulation(params):
    """
    Executa a integração temporal do sistema acoplado LBM-CH.
    Implementa Double Buffering para eliminação de overhead de alocação em tempo de execução.
    """
    start_time = time.time()

    # --- Configuração de Parâmetros Termodinâmicos ---
    sigma = params["SIGMA"]
    int_width = params["INTERFACE_WIDTH"]

    # Ajuste do potencial químico para garantir perfil tanh(x/W) e minimizar correntes espúrias
    params["BETA"] = (3.0 * sigma) / (8.0 * int_width)
    params["KAPPA"] = (3.0 * sigma * int_width) / 4.0
    params["DT_CH"] = 1.0 / params["CH_SUBSTEPS"] if params["CH_SUBSTEPS"] > 0 else 0

    # Determinação da topologia do domínio: mode_m = -1 define periodicidade em X e Y
    is_periodic = (params["mode_m"] == -1)

    # Preparação do ambiente de saída
    base_dir = post_process.setup_output_dir(params["id_caso"])

    # --- Inicialização de Campos (Zero-Allocation) ---
    (f_a, f_b), (phi_a, phi_b), psi, rho, u_x, u_y, K_field, (Fx, Fy, mu_buffer) = initialize_fields(params)

    # Ponteiros dinâmicos para troca de buffers (Double Buffering)
    f_in, f_out = f_a, f_b
    phi_in, phi_out = phi_a, phi_b

    max_iter = params["MAX_ITER"]
    snapshot_steps = params["SNAPSHOT_STEPS"]
    checkpoints = np.linspace(0, max_iter - 1, snapshot_steps, dtype=int)

    mass_history = np.zeros(max_iter, dtype=np.float64)
    curv_history = np.zeros(max_iter, dtype=np.float64)
    time_steps = np.arange(max_iter)

    # --- Loop de Evolução Temporal ---
    for t in tqdm(range(max_iter), desc=f"Integrando: {params['id_caso']}"):

        # 1. Campo Magnético (Poisson/SOR)
        if params["H0"] > 0.0:
            chi_field = np.clip((phi_in + 1.0) * 0.5, 0.0, 1.0) * params["CHI_MAX"]
            psi = solve_poisson_magnetic(psi, chi_field, params["H0"], params["H_ANGLE"], params["SOR_OMEGA"])
        else:
            chi_field = np.zeros_like(phi_in)

        # 2. Evolução da Interface (Cahn-Hilliard)
        if params["CH_SUBSTEPS"] > 0:
            for _ in range(params["CH_SUBSTEPS"]):
                cahn_hilliard_substep(
                    phi_in, phi_out, mu_buffer, u_x, u_y,
                    params["BETA"], params["KAPPA"], params["DT_CH"],
                    params["M_MOBILITY"], is_periodic
                )
                # Swap local para sub-iteração de CH
                phi_in, phi_out = phi_out, phi_in

        # 3. Solução Hidrodinâmica (Lattice Boltzmann D2Q9)
        lbm_step(
            f_in, f_out, phi_in, psi, rho, u_x, u_y, chi_field, K_field, Fx, Fy,
            params["TAU_IN"], params["TAU_OUT"], params["U_INLET"],
            params["BETA"], params["KAPPA"], is_periodic
        )

        # Swap principal de tensores de distribuição de partículas
        f_in, f_out = f_out, f_in

        # 4. Coleta de Dados e Diagnósticos
        mass_history[t] = np.sum(rho)
        if params["CH_SUBSTEPS"] > 0:
            curv_history[t] = post_process.compute_interface_curvature(phi_in)

        if t in checkpoints:
            post_process.export_fields_vtk(phi_in, psi, rho, u_x, u_y, t, base_dir)

    exec_duration = time.time() - start_time

    # --- Finalização e Validação ---
    post_process.export_time_series(mass_history, curv_history, time_steps, base_dir)
    post_process.export_simulation_log(params, mass_history, curv_history, exec_duration, base_dir)

    # Seleção automática da rotina de validação analítica
    if params["CH_SUBSTEPS"] == 0 and params["mode_m"] == 0:
        # Validação da Lei de Darcy (Escoamento em Meio Poroso)
        post_process.validate_darcy_flow(rho, params, base_dir)
    elif params["mode_m"] == -1:
        # Validação de Young-Laplace (Tensão Superficial Estática)
        print(f"\nCaso de Young-Laplace finalizado. Utilize o script 'valida_laplace.py' para análise.")
    else:
        # Exportação de instabilidade de Saffman-Taylor
        post_process.export_tip_position(phi_in, base_dir)

    print(f"\nProcesso concluído para o caso: {params['id_caso']}")
    print(f"Duração total: {exec_duration:.2f} s | Resultados em: {base_dir}\n")


if __name__ == "__main__":
    try:
        with open("casos.json", "r", encoding="utf-8") as f:
            casos_para_rodar = json.load(f)
    except Exception as e:
        print(f"Erro fatal ao carregar configurações: {e}")
        exit(1)

    print(f"{'=' * 80}")
    print(f"SIMULADOR LBM MULTIFÁSICO - VARREDURA PARAMÉTRICA")
    print(f"{'=' * 80}")

    for index, caso in enumerate(casos_para_rodar):
        print(f"Iniciando Instância [{index + 1}/{len(casos_para_rodar)}]: {caso['id_caso']}")
        run_simulation(caso)

    print(f"{'=' * 80}")
    print("VARREDURA CONCLUÍDA COM SUCESSO.")
    print(f"{'=' * 80}")