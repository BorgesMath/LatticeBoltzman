# main.py
import json
import time
import numpy as np
from tqdm import tqdm

# Importação dos kernels e módulos de suporte
from initialization.initialization import initialize_fields
from Magnetismo.poisson import solve_poisson_magnetic
from cahn_hilliard.cahn_hilliard import cahn_hilliard_substep
from lbm.lbm import lbm_step
from post_process import post_process

def run_simulation(params):
    """
    Orquestra a simulação LBM-CH multifásica.
    Focada em estabilidade linear (LSA), validação de Darcy e Young-Laplace.
    """
    start_time = time.time()

    # --- 1. Configuração de Parâmetros Termodinâmicos (CORRIGIDOS) ---
    sigma = params["SIGMA"]
    int_width = params["INTERFACE_WIDTH"]

    # Relações analíticas para perfil tanh(x/W) que garantem erro < 2% em Laplace
    params["BETA"] = (3.0 * sigma) / (8.0 * int_width)
    params["KAPPA"] = (3.0 * sigma * int_width) / 4.0
    params["DT_CH"] = 1.0 / params["CH_SUBSTEPS"] if params["CH_SUBSTEPS"] > 0 else 0

    # Definição da Topologia do Domínio
    # mode_m = -1 aciona o modo 'caixa periódica' para Young-Laplace
    is_periodic = (params["mode_m"] == -1)

    # Preparação do diretório de saída
    base_dir = post_process.setup_output_dir(params["id_caso"])

    # --- 2. Inicialização com Gradiente de Darcy (Pre-pressurização) ---
    # Agora o initialize_fields já retorna o campo 'rho' com o perfil linear dp/dx
    (f_a, f_b), (phi_a, phi_b), psi, rho, u_x, u_y, K_field, (Fx, Fy, mu_buffer) = initialize_fields(params)

    # Ponteiros para Double Buffering
    f_in, f_out = f_a, f_b
    phi_in, phi_out = phi_a, phi_b

    max_iter = params["MAX_ITER"]
    snapshot_steps = params["SNAPSHOT_STEPS"]
    checkpoints = np.linspace(0, max_iter - 1, snapshot_steps, dtype=int)

    # Séries temporais para análise LSA e integridade
    mass_history = np.zeros(max_iter, dtype=np.float64)
    time_steps = np.arange(max_iter)

    # --- Extração do Campo Magnético de Fundo ---
    # Cálculo prévio das componentes macroscópicas invariantes no tempo para otimização
    angle_rad = np.radians(params["H_ANGLE"])
    Hx_fundo = params["H0"] * np.cos(angle_rad)
    Hy_fundo = params["H0"] * np.sin(angle_rad)

    # --- 2.5 Pré-convergência do potencial magnético (warm-up SOR) ---
    # Garante que psi_tilde está convergido antes do primeiro passo:
    # com psi=0 inicial, 15 varreduras só propagam ~15 células no domínio.
    if params["H0"] > 0.0:
        chi_init = np.clip((phi_in + 1.0) * 0.5, 0.0, 1.0) * params["CHI_MAX"]
        for _ in range(30):  # 30 × 15 = 450 varreduras (suficiente p/ NX=1500)
            psi = solve_poisson_magnetic(
                psi, chi_init, Hx_fundo, Hy_fundo, params["SOR_OMEGA"]
            )

    # --- 3. Ciclo de Integração Temporal ---
    for t in tqdm(range(max_iter), desc=f"Integrando: {params['id_caso']}"):

        # A. Solução Magnética (Poisson/SOR)
        if params["H0"] > 0.0:
            chi_field = np.clip((phi_in + 1.0) * 0.5, 0.0, 1.0) * params["CHI_MAX"]
            psi = solve_poisson_magnetic(psi, chi_field, Hx_fundo, Hy_fundo, params["SOR_OMEGA"])
        else:
            chi_field = np.zeros_like(phi_in)

        # B. Dinâmica de Interface (Cahn-Hilliard)
        if params["CH_SUBSTEPS"] > 0:
            for _ in range(params["CH_SUBSTEPS"]):
                cahn_hilliard_substep(
                    phi_in, phi_out, mu_buffer, u_x, u_y,
                    params["BETA"], params["KAPPA"], params["DT_CH"],
                    params["M_MOBILITY"], is_periodic
                )
                phi_in, phi_out = phi_out, phi_in

        # C. Hidrodinâmica e Forças (Lattice Boltzmann D2Q9)
        # Inserção das componentes do campo de fundo na assinatura da função
        lbm_step(
            f_in, f_out, phi_in, psi, rho, u_x, u_y, chi_field, K_field, Fx, Fy,
            params["TAU_IN"], params["TAU_OUT"], params["U_INLET"],
            params["BETA"], params["KAPPA"], is_periodic, Hx_fundo, Hy_fundo
        )

        # Swap de Buffers LBM
        f_in, f_out = f_out, f_in

        # D. Diagnósticos em tempo real
        mass_history[t] = np.sum(rho)

        # E. Exportação VTK (Paraview)
        if t in checkpoints:
            post_process.export_fields_vtk(phi_in, psi, rho, u_x, u_y, t, base_dir,
                                           Hx_fundo, Hy_fundo)

    exec_duration = time.time() - start_time

    # --- 4. Finalização e Exportação de Logs ---
    post_process.export_time_series(mass_history, time_steps, base_dir)
    post_process.export_simulation_log(params, mass_history, exec_duration, base_dir)

    # Seleção de Rotina de Validação Final
    if params["CH_SUBSTEPS"] == 0 and params["mode_m"] == 0:
        post_process.validate_darcy_flow(rho, params, base_dir)
    elif params["mode_m"] == -1:
        print(f"\n[OK] Simulação de Laplace concluída. Use 'valida_laplace.py'.")
    else:
        post_process.export_tip_position(phi_in, base_dir)

    print(f"\nFinalizado: {params['id_caso']} | Tempo: {exec_duration:.2f}s\n")

if __name__ == "__main__":
    try:
        with open("casos.json", "r", encoding="utf-8") as f:
            casos_para_rodar = json.load(f)
    except Exception as e:
        print(f"Erro ao carregar casos.json: {e}")
        exit(1)

    for index, caso in enumerate(casos_para_rodar):
        print(f"Executando {index + 1}/{len(casos_para_rodar)}: {caso['id_caso']}")
        run_simulation(caso)