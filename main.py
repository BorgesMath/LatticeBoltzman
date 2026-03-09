# main.py
import numpy as np
from tqdm import tqdm

from config.config import (MAX_ITER, SNAPSHOT_STEPS, CHI_MAX, CH_SUBSTEPS,
                           METODO_MAGNETISMO, H0, H_ANGLE, NY, NX)
from initialization.initialization import initialize_fields

# Importação dos módulos de Magnetismo
from Magnetismo.poisson import solve_poisson_magnetic
from Magnetismo.paramagnetico import compute_paramagnetic_field_analytical

from cahn_hilliard.cahn_hilliard import cahn_hilliard_substep
from lbm.lbm import lbm_step
from post_process import post_process


def get_interface_amplitude(phi):
    """
    Estima a amplitude 'lambda' da perturbação medindo a extensão
    da interface (diferença entre o pico máximo e o vale mínimo em X).
    """
    # Encontra onde a interface está (phi ~ 0)
    # Como a interface é vertical (varia em Y), olhamos a variação em X.
    interface_mask = np.abs(phi) < 0.1

    if not np.any(interface_mask):
        return 0.0

    # Coordenadas X dos pontos da interface
    _, x_indices = np.where(interface_mask)

    if len(x_indices) == 0:
        return 0.0

    x_max = np.max(x_indices)
    x_min = np.min(x_indices)

    # Amplitude é metade da distância pico-vale
    amplitude = (x_max - x_min) / 2.0
    return amplitude


def run_simulation(mode_m=4, amplitude=2.0):
    """
    Orquestra a simulação da Instabilidade de Saffman-Taylor Magnética.
    """
    base_dir = post_process.setup_output_dir(mode_m)

    # 1. Inicialização dos Campos
    f, phi, psi, rho, u_x, u_y, K_field = initialize_fields(mode_m, amplitude)

    checkpoints = np.linspace(0, MAX_ITER - 1, SNAPSHOT_STEPS, dtype=int)
    mass_history = np.zeros(MAX_ITER, dtype=np.float64)
    curv_history = np.zeros(MAX_ITER, dtype=np.float64)
    time_steps = np.arange(MAX_ITER)

    # 2. Preparação dos Parâmetros para o Método Analítico (Paramagnético)
    # Isso evita recálculos desnecessários dentro do loop
    if METODO_MAGNETISMO == 'PARAMAGNETICO':
        # Conversão do ângulo (0 = Normal/X, 90 = Tangencial/Y)
        rad = np.radians(H_ANGLE)

        # Decomposição do Campo Base
        H0n1 = H0 * np.cos(rad)  # Componente Normal (X) - Principal motor da instabilidade
        H0t = H0 * np.sin(rad)  # Componente Tangencial (Y)

        # Permeabilidades
        mu0 = 1.0
        mu1 = mu0 * (1.0 + CHI_MAX)  # Fluido Invasor (Ferrofluido)
        mu2 = mu0  # Fluido Residente

        # Número de onda (k) para perturbação periódica em Y (Domínio de altura NY)
        k_wave = (2.0 * np.pi * mode_m) / NY

        print(f"--- Configuração Analítica ---")
        print(f"H0 Normal (X): {H0n1:.4f} | H0 Tangencial (Y): {H0t:.4f}")
        print(f"Wavenumber (k): {k_wave:.4f}")
        print(f"------------------------------")

    print(f"Iniciando Simulação. Modo Magnético: {METODO_MAGNETISMO}")

    # 3. Loop de Tempo
    for t in tqdm(range(MAX_ITER), desc=f"Integração (Modo {mode_m})"):

        # --- A. Solver Magnético ---
        # Atualiza o campo de susceptibilidade baseado na posição da interface
        chi_field = np.clip((phi + 1.0) * 0.5, 0.0, 1.0) * CHI_MAX

        if METODO_MAGNETISMO == 'POISSON':  # ou 'NUMERICO'
            # Método Numérico Iterativo (SOR)
            psi = solve_poisson_magnetic(psi, chi_field)

        elif METODO_MAGNETISMO == 'PARAMAGNETICO':
            # Método Analítico (Imposto via PDF)

            # 1. Mede a amplitude atual da perturbação
            current_lambda = get_interface_amplitude(phi)

            # 2. Encontra a posição média X da interface (para centralizar a solução analítica)
            try:
                # Pega índices X onde phi ~ 0
                interface_pts = np.where(np.abs(phi) < 0.1)[1]
                if len(interface_pts) > 0:
                    current_x_interface = np.mean(interface_pts)
                else:
                    current_x_interface = 80.0  # Fallback se a interface sumir
            except:
                current_x_interface = 80.0

            # 3. Chama a função Numba com 10 argumentos explícitos
            psi = compute_paramagnetic_field_analytical(
                psi,
                0,
                NX,  # nx
                NY,  # ny
                k_wave,  # k
                H0t,  # H0t (Tangencial Y)
                H0n1,  # H0n1 (Normal X)
                mu1,
                mu2,
                current_x_interface  # Posição central da interface em X
            )

        # --- B. Dinâmica Interfacial (Cahn-Hilliard) ---
        for _ in range(CH_SUBSTEPS):
            phi = cahn_hilliard_substep(phi, u_x, u_y)

        # --- C. Hidrodinâmica (LBM) ---
        f, rho, u_x, u_y = lbm_step(f, phi, psi, rho, u_x, u_y, chi_field, K_field)

        # --- D. Diagnósticos e Exportação ---
        mass_history[t] = np.sum(rho)
        curv_history[t] = post_process.compute_interface_curvature(phi)

        if t in checkpoints:
            post_process.export_fields(phi, psi, rho, u_x, u_y, mode_m, t, base_dir)

    # 4. Exportação Final
    post_process.export_time_series(mass_history, curv_history, time_steps, mode_m, base_dir)
    post_process.export_tip_position(phi, mode_m, base_dir)
    print(f"\nIntegração concluída. Dados em: {base_dir}")


if __name__ == "__main__":
    # Ajuste aqui o modo inicial e a amplitude da perturbação
    run_simulation(mode_m=4, amplitude=2.0)