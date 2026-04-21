# main.py
import json
import numpy as np
from tqdm import tqdm

from initialization.initialization import initialize_fields
from Magnetismo.poisson import solve_poisson_magnetic
from cahn_hilliard.cahn_hilliard import cahn_hilliard_substep
from lbm.lbm import lbm_step
from post_process import post_process

def run_simulation(params):
    """
    Orquestra a simulação com injeção de parâmetros via dicionário.
    """
    # Desempacotamento numérico
    id_caso = params["id_caso"]
    mode_m = params["mode_m"]
    amplitude = params["amplitude"]
    max_iter = params["MAX_ITER"]
    snapshot_steps = params["SNAPSHOT_STEPS"]
    chi_max = params["CHI_MAX"]
    ch_substeps = params["CH_SUBSTEPS"]

    base_dir = post_process.setup_output_dir(id_caso) # Agora usamos id_caso para criar a pasta

    # Alocação
    f, phi, psi, rho, u_x, u_y, K_field = initialize_fields(mode_m, amplitude)

    checkpoints = np.linspace(0, max_iter - 1, snapshot_steps, dtype=int)
    mass_history = np.zeros(max_iter, dtype=np.float64)
    curv_history = np.zeros(max_iter, dtype=np.float64)
    time_steps = np.arange(max_iter)

    # Integração Temporal Numérica
    for t in tqdm(range(max_iter), desc=f"Integração ({id_caso})"):

        chi_field = np.clip((phi + 1.0) * 0.5, 0.0, 1.0) * chi_max
        psi = solve_poisson_magnetic(psi, chi_field)

        for _ in range(ch_substeps):
            phi = cahn_hilliard_substep(phi, u_x, u_y)

        f, rho, u_x, u_y = lbm_step(f, phi, psi, rho, u_x, u_y, chi_field, K_field)

        # Diagnósticos
        mass_history[t] = np.sum(rho)
        curv_history[t] = post_process.compute_interface_curvature(phi)

        # Exportação de Campos
        if t in checkpoints:
            post_process.export_fields_vtk(phi, psi, rho, u_x, u_y, mode_m, t, base_dir)

    # Exportação Final
    post_process.export_time_series(mass_history, curv_history, time_steps, mode_m, base_dir)
    post_process.export_tip_position(phi, mode_m, base_dir)

    print(f"\nIntegração concluída. Diagnósticos em: {base_dir}")


if __name__ == "__main__":
    # Carregamento do vetor de casos paramétricos
    try:
        with open("casos.json", "r", encoding="utf-8") as file:
            lista_casos = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError("Arquivo 'casos.json' não encontrado no diretório raiz.")

    # Loop de orquestração Batch
    for caso in lista_casos:
        print(f"\n{'='*60}\nIniciando rotina para: {caso['id_caso']}\n{'='*60}")
        run_simulation(caso)