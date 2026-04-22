# main.py
import json
import time
import numpy as np
from tqdm import tqdm

# Importação dos módulos do núcleo de cálculo refatorados
from initialization.initialization import initialize_fields
from Magnetismo.poisson import solve_poisson_magnetic
from cahn_hilliard.cahn_hilliard import cahn_hilliard_substep
from lbm.lbm import lbm_step
from post_process import post_process


def run_simulation(params):
    """
    Orquestra a simulação Lattice Boltzmann acoplada.
    Gerencia a alternância de ponteiros (Double Buffering) e injeção dinâmica de parâmetros.
    """
    start_time = time.time()

    # --- Derivação Termodinâmica de Ginzburg-Landau (ESTRITAMENTE CORRIGIDA) ---
    sigma = params["SIGMA"]
    int_width = params["INTERFACE_WIDTH"]
    # BETA deve ser inversamente proporcional à largura para manter a espessura de equilíbrio
    params["BETA"] = (3.0 * sigma) / (4.0 * int_width)
    params["KAPPA"] = (3.0 * sigma * int_width) / 8.0
    params["DT_CH"] = 1.0 / params["CH_SUBSTEPS"] if params["CH_SUBSTEPS"] > 0 else 0

    # Configuração do diretório de saída com ID e Timestamp (Windows Compatible)
    base_dir = post_process.setup_output_dir(params["id_caso"])

    # --- Inicialização Zero-Allocation ---
    # Desempacotamento dos 8 tensores e buffers pre-alocados
    (f_a, f_b), (phi_a, phi_b), psi, rho, u_x, u_y, K_field, (Fx, Fy, mu_buffer) = initialize_fields(params)

    # Definição dos ponteiros de leitura (in) e escrita (out) para o Double Buffering
    f_in, f_out = f_a, f_b
    phi_in, phi_out = phi_a, phi_b

    max_iter = params["MAX_ITER"]
    snapshot_steps = params["SNAPSHOT_STEPS"]
    checkpoints = np.linspace(0, max_iter - 1, snapshot_steps, dtype=int)

    mass_history = np.zeros(max_iter, dtype=np.float64)
    curv_history = np.zeros(max_iter, dtype=np.float64)
    time_steps = np.arange(max_iter)

    # --- Marcha no Tempo (Integração Numérica) ---
    for t in tqdm(range(max_iter), desc=f"Integrando: {params['id_caso']}"):

        # 1. Solução do Campo Magnético (Poisson/SOR)
        if params["H0"] > 0.0:
            chi_field = np.clip((phi_in + 1.0) * 0.5, 0.0, 1.0) * params["CHI_MAX"]
            psi = solve_poisson_magnetic(psi, chi_field, params["H0"], params["H_ANGLE"], params["SOR_OMEGA"])
        else:
            chi_field = np.zeros_like(phi_in)

        # 2. Dinâmica de Interface (Cahn-Hilliard Sub-passos)
        # Omitido se CH_SUBSTEPS == 0 (Casos de Validação de Darcy)
        if params["CH_SUBSTEPS"] > 0:
            for _ in range(params["CH_SUBSTEPS"]):
                cahn_hilliard_substep(
                    phi_in, phi_out, mu_buffer, u_x, u_y,
                    params["BETA"], params["KAPPA"], params["DT_CH"], params["M_MOBILITY"]
                )
                # Swap local de Cahn-Hilliard (Double Buffering)
                phi_in, phi_out = phi_out, phi_in

        # 3. Hidrodinâmica em Meio Poroso (Lattice Boltzmann D2Q9)
        # Kernel agora utiliza Zou-He no Inlet e Força de Kelvin via Hessiano
        lbm_step(
            f_in, f_out, phi_in, psi, rho, u_x, u_y, chi_field, K_field, Fx, Fy,
            params["TAU_IN"], params["TAU_OUT"], params["U_INLET"],
            params["BETA"], params["KAPPA"]
        )

        # Swap principal de ponteiros LBM (O estado t+1 vira a entrada para t+2)
        f_in, f_out = f_out, f_in

        # 4. Coleta de Diagnósticos In-Situ
        mass_history[t] = np.sum(rho)
        if params["CH_SUBSTEPS"] > 0:
            curv_history[t] = post_process.compute_interface_curvature(phi_in)

        # 5. Exportação de Resultados VTK (ParaView)
        if t in checkpoints:
            post_process.export_fields_vtk(phi_in, psi, rho, u_x, u_y, t, base_dir)

    exec_duration = time.time() - start_time

    # --- Exportação Final e Relatórios ---
    post_process.export_time_series(mass_history, curv_history, time_steps, base_dir)
    post_process.export_simulation_log(params, mass_history, curv_history, exec_duration, base_dir)

    # Lógica de Gatilho para Validação Analítica
    if params["CH_SUBSTEPS"] == 0 and params["mode_m"] == 0:
        # Executa a verificação da Lei de Darcy para o escoamento unifásico
        post_process.validate_darcy_flow(rho, params, base_dir)
    else:
        # Exportação padrão para casos de instabilidade interfacial
        post_process.export_tip_position(phi_in, base_dir)

    print(f"\nIntegração concluída para: {params['id_caso']}")
    print(f"Tempo total de execução: {exec_duration:.2f} segundos.")
    print(f"Dados salvos em: {base_dir}\n")


if __name__ == "__main__":
    # Carregamento do Vetor de Casos Paramétricos
    try:
        with open("casos.json", "r", encoding="utf-8") as f:
            casos_para_rodar = json.load(f)
    except FileNotFoundError:
        print("ERRO: O arquivo 'casos.json' não foi encontrado no diretório raiz.")
        exit(1)
    except json.JSONDecodeError:
        print("ERRO: Falha crítica na sintaxe do arquivo 'casos.json'.")
        exit(1)

    print(f"{'=' * 70}")
    print(f"INICIANDO VARREDURA PARAMÉTRICA: {len(casos_para_rodar)} CASO(S) DETECTADO(S)")
    print(f"{'=' * 70}")

    for index, caso in enumerate(casos_para_rodar):
        print(f"Executando Instância [{index + 1}/{len(casos_para_rodar)}]: {caso['id_caso']}")
        run_simulation(caso)

    print(f"{'=' * 70}")
    print("ROTINA DE CÁLCULO FINALIZADA.")
    print(f"{'=' * 70}")