# main.py
import json
import time
import numpy as np
from tqdm import tqdm

# Importação dos módulos do núcleo de cálculo
from initialization.initialization import initialize_fields
from Magnetismo.poisson import solve_poisson_magnetic
from cahn_hilliard.cahn_hilliard import cahn_hilliard_substep
from lbm.lbm import lbm_step
from post_process import post_process


def run_simulation(params):
    """
    Executa uma instância completa da simulação com base no dicionário de parâmetros.
    Gerencia o ciclo de vida: inicialização, integração e exportação de diagnósticos.
    """
    start_time = time.time()

    # --- Derivação de Variáveis Termodinâmicas e Numéricas ---
    # Parâmetros derivados da teoria de campo difuso (Cahn-Hilliard)
    sigma = params["SIGMA"]
    int_width = params["INTERFACE_WIDTH"]
    params["BETA"] = 3.0 * sigma * int_width / 4.0
    params["KAPPA"] = 3.0 * sigma * int_width / 8.0
    params["DT_CH"] = 1.0 / params["CH_SUBSTEPS"]

    # Configuração do diretório de saída com ID do caso e Timestamp (Windows Compatible)
    base_dir = post_process.setup_output_dir(params["id_caso"])

    # --- Inicialização dos Tensores de Campo ---
    f, phi, psi, rho, u_x, u_y, K_field = initialize_fields(params)

    # --- Preparação de Diagnósticos e Checkpoints ---
    max_iter = params["MAX_ITER"]
    snapshot_steps = params["SNAPSHOT_STEPS"]
    checkpoints = np.linspace(0, max_iter - 1, snapshot_steps, dtype=int)

    mass_history = np.zeros(max_iter, dtype=np.float64)
    curv_history = np.zeros(max_iter, dtype=np.float64)
    time_steps = np.arange(max_iter)

    # --- Loop de Integração Temporal (Marcha no Tempo) ---
    for t in tqdm(range(max_iter), desc=f"Integrando: {params['id_caso']}"):

        # 1. Solução do Campo Magnético (Poisson/SOR)
        # O campo de susceptibilidade é mapeado a partir do campo de fase phi
        chi_field = np.clip((phi + 1.0) * 0.5, 0.0, 1.0) * params["CHI_MAX"]
        psi = solve_poisson_magnetic(
            psi, chi_field, params["H0"], params["H_ANGLE"], params["SOR_OMEGA"]
        )

        # 2. Dinâmica de Interface (Cahn-Hilliard)
        # Sub-passos para garantir a estabilidade da interface difusa
        for _ in range(params["CH_SUBSTEPS"]):
            phi = cahn_hilliard_substep(
                phi, u_x, u_y, params["BETA"], params["KAPPA"],
                params["DT_CH"], params["M_MOBILITY"]
            )

        # 3. Hidrodinâmica em Meio Poroso (Lattice Boltzmann - D2Q9)
        # Acoplamento Força de Darcy-Brinkman + Tensão Superficial + Força Magnética
        f, rho, u_x, u_y = lbm_step(
            f, phi, psi, rho, u_x, u_y, chi_field, K_field,
            params["TAU_IN"], params["TAU_OUT"], params["U_INLET"],
            params["BETA"], params["KAPPA"]
        )

        # 4. Coleta de Invariantes e Diagnósticos
        mass_history[t] = np.sum(rho)
        curv_history[t] = post_process.compute_interface_curvature(phi)

        # 5. Exportação Binária de Campos (VTK para ParaView)
        if t in checkpoints:
            post_process.export_fields_vtk(phi, psi, rho, u_x, u_y, t, base_dir)

    # --- Finalização e Exportação de Resultados ---
    end_time = time.time()
    exec_duration = end_time - start_time

    # Séries temporais (PNG), Posição da Ponta (TXT) e Log de Integridade (JSON)
    post_process.export_time_series(mass_history, curv_history, time_steps, base_dir)
    post_process.export_tip_position(phi, base_dir)
    post_process.export_simulation_log(params, mass_history, curv_history, exec_duration, base_dir)

    print(f"\nFinalizado: {params['id_caso']} | Tempo: {exec_duration:.2f}s")
    print(f"Diretório de saída: {base_dir}\n")


if __name__ == "__main__":
    # Carregamento do arquivo de configuração JSON
    try:
        with open("casos.json", "r", encoding="utf-8") as f:
            casos_para_rodar = json.load(f)
    except FileNotFoundError:
        print("ERRO: O arquivo 'casos.json' não foi encontrado no diretório raiz.")
        exit(1)
    except json.JSONDecodeError:
        print("ERRO: Falha na formatação do arquivo 'casos.json'. Verifique a sintaxe.")
        exit(1)

    # Execução sequencial de todos os casos definidos
    print(f"{'=' * 70}")
    print(f"INICIANDO VARREDURA PARAMÉTRICA: {len(casos_para_rodar)} CASO(S) DETECTADO(S)")
    print(f"{'=' * 70}")

    for index, caso in enumerate(casos_para_rodar):
        print(f"Rodando Caso [{index + 1}/{len(casos_para_rodar)}]: {caso['id_caso']}")
        run_simulation(caso)

    print(f"{'=' * 70}")
    print("VARREDURA CONCLUÍDA COM SUCESSO.")
    print(f"{'=' * 70}")