# main.py
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from config.config import (MAX_ITER, SNAPSHOT_STEPS, CHI_MAX, CH_SUBSTEPS,
                           METODO_MAGNETISMO, H0, H_ANGLE, NY, NX)
from initialization.initialization import initialize_fields

# Importação dos módulos de Magnetismo
from Magnetismo.poisson import solve_poisson_magnetic
from Magnetismo.paramagnetico import compute_paramagnetic_field_analytical

from Multifasico.cahn_hilliard import cahn_hilliard_substep
from lbm.lbm import lbm_step
from post_process import post_process
from lsa.lsa import analyze_stability


def get_interface_amplitude_espacial(phi):
    """
    Mtodo 1: Rastreio de pico/vale sub-pixel (Métrica Geométrica Clássica).
    Mede a distância escalar entre a crista mais avançada e o vale mais recuado.
    """
    ny, nx = phi.shape
    x_positions = []

    for y in range(ny):
        crossings = np.where(np.diff(np.sign(phi[y, :])))[0]
        if len(crossings) > 0:
            x_idx = crossings[-1]  # Último cruzamento (frente de invasão)
            phi0 = phi[y, x_idx]
            phi1 = phi[y, x_idx + 1]

            if np.abs(phi1 - phi0) > 1e-6:
                x_subpixel = x_idx - phi0 / (phi1 - phi0)
                x_positions.append(x_subpixel)

    if not x_positions:
        return 0.0

    return (np.max(x_positions) - np.min(x_positions)) / 2.0


def get_amplitude_fft(phi, mode_m):
    """
    Mtodo 2: Isolamento Espectral (Transformada Rápida de Fourier).
    Extrai o perfil h(y), aplica a FFT e retorna estritamente a amplitude
    do coeficiente associado ao número de onda instigado.
    """
    ny_shape, nx_shape = phi.shape
    h_y = np.zeros(ny_shape, dtype=np.float64)

    for y in range(ny_shape):
        crossings = np.where(np.diff(np.sign(phi[y, :])))[0]
        if len(crossings) > 0:
            x_idx = crossings[-1]
            phi0 = phi[y, x_idx]
            phi1 = phi[y, x_idx + 1]
            if np.abs(phi1 - phi0) > 1e-6:
                h_y[y] = x_idx - phi0 / (phi1 - phi0)
            else:
                h_y[y] = x_idx
        else:
            h_y[y] = np.mean(h_y[:y]) if y > 0 else nx_shape / 2.0

    h_y_centrado = h_y - np.mean(h_y)
    espectro = np.fft.rfft(h_y_centrado)

    # Normalização pela malha e conversão para amplitude física
    amplitude_espectral = 2.0 * np.abs(espectro[mode_m]) / ny_shape
    return amplitude_espectral


def run_simulation(mode_m=32, amplitude=2.0):
    base_dir = post_process.setup_output_dir(mode_m)

    # 1. Cálculo da Taxa Teórica Analítica
    s_teorico = analyze_stability(mode_m, base_dir)

    # 2. Inicialização dos Campos
    f, phi, psi, rho, u_x, u_y, K_field = initialize_fields(mode_m, amplitude)

    checkpoints = np.linspace(0, MAX_ITER - 1, SNAPSHOT_STEPS, dtype=int)
    mass_history = np.zeros(MAX_ITER, dtype=np.float64)
    curv_history = np.zeros(MAX_ITER, dtype=np.float64)

    # Vetores de Histórico Duplos
    amp_hist_espacial = np.zeros(MAX_ITER, dtype=np.float64)
    amp_hist_fft = np.zeros(MAX_ITER, dtype=np.float64)
    time_steps = np.arange(MAX_ITER)

    # Configuração de Solver Magnético
    if METODO_MAGNETISMO == 'PARAMAGNETICO':
        rad = np.radians(H_ANGLE)
        H0n1 = H0 * np.cos(rad)
        H0t = H0 * np.sin(rad)
        mu1 = 1.0 * (1.0 + CHI_MAX)
        mu2 = 1.0
        k_wave = (2.0 * np.pi * mode_m) / NY

    print(f"Iniciando Simulação LBM. Magnético: {METODO_MAGNETISMO} | LSA Mode: {mode_m}")

    # 3. Integração Temporal
    pbar = tqdm(range(MAX_ITER), desc=f"Integração")

    for t in pbar:
        # Extração Bifurcada das Amplitudes
        amp_espacial = get_interface_amplitude_espacial(phi)
        amp_fft = get_amplitude_fft(phi, mode_m)

        amp_hist_espacial[t] = amp_espacial
        amp_hist_fft[t] = amp_fft

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
                psi, amp_fft, NX, NY, k_wave, H0t, H0n1, mu1, mu2, current_x_interface
            )

        # --- Cahn-Hilliard & LBM ---
        for _ in range(CH_SUBSTEPS):
            phi = cahn_hilliard_substep(phi, u_x, u_y)
        f, rho, u_x, u_y = lbm_step(f, phi, psi, rho, u_x, u_y, chi_field, K_field)

        # --- Diagnósticos de Massa e Curvatura ---
        mass_history[t] = np.sum(rho)
        curv_history[t] = post_process.compute_interface_curvature(phi)

        # --- Regressão Linear Dupla em Tempo Real (Pós-Transiente) ---
        if t > 100 and t % 50 == 0:
            idx_start = max(0, t - 600)
            t_win = time_steps[idx_start:t]

            # Filtro da Janela Espacial
            win_esp = amp_hist_espacial[idx_start:t]
            mask_esp = win_esp > 1e-4
            s_esp = 0.0
            if np.sum(mask_esp) > 10:
                s_esp, _ = np.polyfit(t_win[mask_esp], np.log(win_esp[mask_esp]), 1)

            # Filtro da Janela FFT
            win_fft = amp_hist_fft[idx_start:t]
            mask_fft = win_fft > 1e-4
            s_fft = 0.0
            if np.sum(mask_fft) > 10:
                s_fft, _ = np.polyfit(t_win[mask_fft], np.log(win_fft[mask_fft]), 1)

            err_fft = abs(s_fft - s_teorico) / abs(s_teorico) * 100 if s_teorico != 0 else 0.0

            pbar.set_postfix({
                's_Espacial': f"{s_esp:.2e}",
                's_FFT': f"{s_fft:.2e}",
                's_Teoria': f"{s_teorico:.2e}",
                'Err_FFT': f"{err_fft:.1f}%"
            })

        if t in checkpoints:
            post_process.export_fields(phi, psi, rho, u_x, u_y, mode_m, t, base_dir)

    # 4. Processamento Topológico Consolidado
    post_process.export_time_series(mass_history, curv_history, time_steps, mode_m, base_dir)
    post_process.export_tip_position(phi, mode_m, base_dir)

    # Exporta a validação utilizando a métrica primária (FFT por ter maior rigor)
    post_process.export_growth_rate(amp_hist_fft, time_steps, mode_m, s_teorico, base_dir)

    # 5. Diagramação Comparativa dos Dois Métodos
    _plot_comparativo_metodos(time_steps, amp_hist_espacial, amp_hist_fft, s_teorico, mode_m, base_dir)


def _plot_comparativo_metodos(t_arr, amp_esp, amp_fft, s_teo, mode_m, base_dir):
    """
    Gera curva logarítmica comparando a precisão LSA do rastreio de interface
    frente à extração espectral.
    """
    plt.figure(figsize=(10, 6))

    idx_10 = int(len(t_arr) * 0.10)
    t_val = t_arr[idx_10:]

    # Processamento Espacial
    esp_val = amp_esp[idx_10:]
    mask_esp = esp_val > 1e-4
    if np.any(mask_esp):
        ln_esp = np.log(esp_val[mask_esp] / esp_val[mask_esp][0])
        t_esp = t_val[mask_esp] - t_val[mask_esp][0]
        plt.plot(t_esp, ln_esp, 'b-', alpha=0.7, linewidth=2, label='Geométrico/Espacial')
        s_num_esp, _ = np.polyfit(t_esp[:int(len(t_esp) * 0.33)], ln_esp[:int(len(t_esp) * 0.33)], 1)
    else:
        s_num_esp = 0.0

    # Processamento FFT
    fft_val = amp_fft[idx_10:]
    mask_fft = fft_val > 1e-4
    if np.any(mask_fft):
        ln_fft = np.log(fft_val[mask_fft] / fft_val[mask_fft][0])
        t_fft = t_val[mask_fft] - t_val[mask_fft][0]
        plt.plot(t_fft, ln_fft, 'g-', linewidth=2.5, label='Espectral/FFT')
        s_num_fft, _ = np.polyfit(t_fft[:int(len(t_fft) * 0.33)], ln_fft[:int(len(t_fft) * 0.33)], 1)
    else:
        s_num_fft = 0.0

    # Reta Analítica
    plt.plot(t_val - t_val[0], s_teo * (t_val - t_val[0]), 'r--', linewidth=2, label=f'Analítico (s={s_teo:.2e})')

    plt.title(f"Convergência de Métodos de Extração LSA (Modo {mode_m})\n"
              f"Erro Relativo Espacial: {abs(s_num_esp - s_teo) / abs(s_teo) * 100 if s_teo != 0 else 0:.2f}% | "
              f"Erro Relativo FFT: {abs(s_num_fft - s_teo) / abs(s_teo) * 100 if s_teo != 0 else 0:.2f}%")

    plt.xlabel(r'Tempo Transladado $t - t_{valid}$')
    plt.ylabel(r'$\ln(\lambda / \lambda_0)$')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'series_temporais', f'comparativo_metodos_lsa_modo_{mode_m}.png'), dpi=200)
    plt.close()


if __name__ == "__main__":
    run_simulation(mode_m=4, amplitude=2.0)