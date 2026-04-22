# main.py
import json
import time
import numpy as np
from tqdm import tqdm

from initialization.initialization import initialize_fields
from Magnetismo.poisson import solve_poisson_magnetic
from cahn_hilliard.cahn_hilliard import cahn_hilliard_substep
from lbm.lbm import lbm_step
from post_process import post_process


def run_simulation(params):
    start_time = time.time()

    # --- Derivação Termodinâmica ---
    sigma = params["SIGMA"]
    int_width = params["INTERFACE_WIDTH"]
    params["BETA"] = (3.0 * sigma) / (8.0 * int_width)
    params["KAPPA"] = (3.0 * sigma * int_width) / 4.0
    #params["BETA"] = (3.0 * sigma) / (4.0 * int_width)
    #params["KAPPA"] = (3.0 * sigma * int_width) / 8.0
    params["DT_CH"] = 1.0 / params["CH_SUBSTEPS"]

    base_dir = post_process.setup_output_dir(params["id_caso"])

    # --- Inicialização (Double Buffering) ---
    # Desempacotamento correto dos 8 conjuntos de retorno
    (f_a, f_b), (phi_a, phi_b), psi, rho, u_x, u_y, K_field, (Fx, Fy, mu_buffer) = initialize_fields(params)

    # Definição inicial dos ponteiros de leitura (in) e escrita (out)
    f_in, f_out = f_a, f_b
    phi_in, phi_out = phi_a, phi_b

    max_iter = params["MAX_ITER"]
    snapshot_steps = params["SNAPSHOT_STEPS"]
    checkpoints = np.linspace(0, max_iter - 1, snapshot_steps, dtype=int)

    mass_history = np.zeros(max_iter, dtype=np.float64)
    curv_history = np.zeros(max_iter, dtype=np.float64)
    time_steps = np.arange(max_iter)

    # --- Marcha no Tempo ---
    for t in tqdm(range(max_iter), desc=f"Integrando: {params['id_caso']}"):

        # 1. Poisson Magnético
        chi_field = np.clip((phi_in + 1.0) * 0.5, 0.0, 1.0) * params["CHI_MAX"]
        psi = solve_poisson_magnetic(psi, chi_field, params["H0"], params["H_ANGLE"], params["SOR_OMEGA"])

        # 2. Cahn-Hilliard Sub-passos
        for _ in range(params["CH_SUBSTEPS"]):
            cahn_hilliard_substep(
                phi_in, phi_out, mu_buffer, u_x, u_y,
                params["BETA"], params["KAPPA"], params["DT_CH"], params["M_MOBILITY"]
            )
            # Swap: O resultado (out) torna-se a entrada (in) para o próximo sub-passo
            phi_in, phi_out = phi_out, phi_in

        # 3. Lattice Boltzmann - Meio Poroso e Forças
        lbm_step(
            f_in, f_out, phi_in, psi, rho, u_x, u_y, chi_field, K_field, Fx, Fy,
            params["TAU_IN"], params["TAU_OUT"], params["U_INLET"],
            params["BETA"], params["KAPPA"]
        )
        # Swap: O estado avançado do fluido torna-se o estado atual
        f_in, f_out = f_out, f_in

        # 4. Diagnósticos e Exportação
        mass_history[t] = np.sum(rho)
        curv_history[t] = post_process.compute_interface_curvature(phi_in)

        if t in checkpoints:
            post_process.export_fields_vtk(phi_in, psi, rho, u_x, u_y, t, base_dir)

    exec_duration = time.time() - start_time

    # Exportação Final
    post_process.export_time_series(mass_history, curv_history, time_steps, base_dir)
    post_process.export_tip_position(phi_in, base_dir)
    post_process.export_simulation_log(params, mass_history, curv_history, exec_duration, base_dir)

    print(f"\nFinalizado: {params['id_caso']} | Tempo: {exec_duration:.2f}s")
    print(f"Diretório de saída: {base_dir}\n")


if __name__ == "__main__":
    try:
        with open("casos.json", "r", encoding="utf-8") as f:
            casos_para_rodar = json.load(f)
    except FileNotFoundError:
        print("ERRO: O arquivo 'casos.json' não foi encontrado no diretório raiz.")
        exit(1)
    except json.JSONDecodeError:
        print("ERRO: Falha na formatação do arquivo 'casos.json'.")
        exit(1)

    print(f"{'=' * 70}")
    print(f"INICIANDO VARREDURA PARAMÉTRICA: {len(casos_para_rodar)} CASO(S) DETECTADO(S)")
    print(f"{'=' * 70}")

    for index, caso in enumerate(casos_para_rodar):
        print(f"Rodando Caso [{index + 1}/{len(casos_para_rodar)}]: {caso['id_caso']}")
        run_simulation(caso)

    print(f"{'=' * 70}")
    print("VARREDURA CONCLUÍDA.")
    print(f"{'=' * 70}")