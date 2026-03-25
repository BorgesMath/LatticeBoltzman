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
from lsa.lsa import analyze_stability  # NOVO IMPORT


def get_interface_amplitude(phi):
    """
    Mede a amplitude da interface de forma imune à espessura de Cahn-Hilliard.
    Localiza a travessia exata do zero (phi=0) em cada linha horizontal (Y).
    """
    ny, nx = phi.shape
    x_positions = []

    for y in range(ny):
        # Encontra onde o sinal inverte (fluido 1 para fluido 2)
        crossings = np.where(np.diff(np.sign(phi[y, :])))[0]
        if len(crossings) > 0:
            x_idx = crossings[0]
            # Interpolação linear sub-pixel para precisão máxima
            # x_real = x0 - phi(x0) * (x1 - x0) / (phi(x1) - phi(x0))
            phi0 = phi[y, x_idx]
            phi1 = phi[y, x_idx + 1]

            # Evita divisão por zero
            if np.abs(phi1 - phi0) > 1e-6:
                x_subpixel = x_idx - phi0 / (phi1 - phi0)
                x_positions.append(x_subpixel)

    if not x_positions:
        return 0.0

    # Amplitude = (Max X - Min X) / 2
    return (np.max(x_positions) - np.min(x_positions)) / 2.0


def run_simulation(mode_m=4, amplitude=2.0):
    base_dir = post_process.setup_output_dir(mode_m)

    # 1. Cálculo da Taxa Teórica (LSA)
    s_teorico = analyze_stability(mode_m, base_dir)

    # 2. Inicialização
    f, phi, psi, rho, u_x, u_y, K_field = initialize_fields(mode_m, amplitude)

    checkpoints = np.linspace(0, MAX_ITER - 1, SNAPSHOT_STEPS, dtype=int)
    mass_history = np.zeros(MAX_ITER, dtype=np.float64)
    curv_history = np.zeros(MAX_ITER, dtype=np.float64)
    amp_history = np.zeros(MAX_ITER, dtype=np.float64)  # NOVO: Histórico de Amplitude
    time_steps = np.arange(MAX_ITER)

    # Preparação Analítica
    if METODO_MAGNETISMO == 'PARAMAGNETICO':
        rad = np.radians(H_ANGLE)
        H0n1 = H0 * np.cos(rad)
        H0t = H0 * np.sin(rad)
        mu1 = 1.0 * (1.0 + CHI_MAX)
        mu2 = 1.0
        k_wave = (2.0 * np.pi * mode_m) / NY

    print(f"Iniciando Simulação. Modo Magnético: {METODO_MAGNETISMO}")

    # 3. Integração Temporal
    for t in tqdm(range(MAX_ITER), desc=f"Integração (Modo {mode_m})"):

        # Medição Contínua da Amplitude (Fundamental para a validação)
        current_lambda = get_interface_amplitude(phi)
        amp_history[t] = current_lambda

        # --- Solver Magnético ---
        chi_field = np.clip((phi + 1.0) * 0.5, 0.0, 1.0) * CHI_MAX

        if METODO_MAGNETISMO == 'POISSON':
            psi = solve_poisson_magnetic(psi, chi_field)

        elif METODO_MAGNETISMO == 'PARAMAGNETICO':
            try:
                interface_pts = np.where(np.abs(phi) < 0.1)[1]
                current_x_interface = np.mean(interface_pts) if len(interface_pts) > 0 else 80.0
            except:
                current_x_interface = 80.0

            psi = compute_paramagnetic_field_analytical(
                psi, current_lambda, NX, NY, k_wave,
                H0t, H0n1, mu1, mu2, current_x_interface
            )

        # --- Cahn-Hilliard ---
        for _ in range(CH_SUBSTEPS):
            phi = cahn_hilliard_substep(phi, u_x, u_y)

        # --- LBM ---
        f, rho, u_x, u_y = lbm_step(f, phi, psi, rho, u_x, u_y, chi_field, K_field)

        # --- Diagnósticos ---
        mass_history[t] = np.sum(rho)
        curv_history[t] = post_process.compute_interface_curvature(phi)

        if t in checkpoints:
            post_process.export_fields(phi, psi, rho, u_x, u_y, mode_m, t, base_dir)

    # 4. Exportação
    post_process.export_time_series(mass_history, curv_history, time_steps, mode_m, base_dir)
    post_process.export_tip_position(phi, mode_m, base_dir)

    # NOVO: Plota a validação de crescimento
    post_process.export_growth_rate(amp_history, time_steps, mode_m, s_teorico, base_dir)

    print(f"\nIntegração concluída. Dados em: {base_dir}")


if __name__ == "__main__":
    run_simulation(mode_m=8, amplitude=4.0)