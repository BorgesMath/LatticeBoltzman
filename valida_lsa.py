import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('default')
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})


def calc_theoretical_growth_rate(params):
    """
    Transcrição estrita da Equação 26 do documento 'Regime Linear (LSA).tex'
    para as variáveis em unidades de rede (Lattice Units).
    """
    U = params["U_INLET"]
    K = params["K_0"]
    sigma = params["SIGMA"]
    H0 = params["H0"]
    angle = np.radians(params["H_ANGLE"])
    chi_max = params["CHI_MAX"]

    # 1. Parâmetros Viscosos (Viscosidade Dinâmica LBM assumindo rho=1)
    mu_1 = (params["TAU_IN"] - 0.5) / 3.0
    mu_2 = (params["TAU_OUT"] - 0.5) / 3.0

    # 2. Número de Onda (k ou alpha)
    alpha = (2.0 * np.pi * params["mode_m"]) / params["NY"]

    # 3. Magnetismo (mu_m = 1 + chi)
    mu_m1 = 1.0 + chi_max
    mu_m2 = 1.0
    mu_0 = 1.0  # Permeabilidade do vácuo em LU

    # Componentes Normal e Tangencial (Em LBM o fluxo base está em X)
    # Portanto, a normal à interface perturbada é o eixo X.
    H0n = H0 * np.cos(angle)
    H0t = H0 * np.sin(angle)

    # ========================================================
    # Equação 26 - Separação dos Termos
    # ========================================================
    denominador = (mu_1 / K) + (mu_2 / K)

    termo_viscoso = (U / K) * (mu_2 - mu_1)
    termo_gravidade = 0.0  # LBM não tem flutuabilidade no eixo Y atualmente
    termo_capilar = - sigma * (alpha ** 2)

    termo_magnetico = 0.0
    if H0 > 0.0:
        contraste_mag = (mu_m1 - mu_m2) / (mu_m1 + mu_m2)
        termo_magnetico = mu_0 * chi_max * alpha * contraste_mag * (H0n ** 2 - H0t ** 2)

    omega_theo = (alpha / denominador) * (termo_viscoso + termo_gravidade + termo_capilar + termo_magnetico)

    return omega_theo


def process_lsa():
    base_dir = os.getcwd()
    results_dirs = glob.glob(os.path.join(base_dir, "*_d*mes*-h*_min*"))  # Padrão do seu setup_output_dir

    for folder in results_dirs:
        log_path = os.path.join(folder, "relatorio_execucao.json")
        amp_path = os.path.join(folder, "series_temporais", "amplitude.npy")

        if not os.path.exists(log_path) or not os.path.exists(amp_path):
            continue

        with open(log_path, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
        params = log_data["parametros_contorno"]

        amp_history = np.load(amp_path)
        time_steps = np.arange(len(amp_history))

        # 1. Taxa Teórica
        omega_theo = calc_theoretical_growth_rate(params)

        # 2. Extração Numérica (Isolando o Regime Linear)
        # Descarta o transiente inicial (ex: primeiras 5% iterações) para a pressão estabilizar
        # e corta quando a amplitude ultrapassar 10% do domínio (início do regime não-linear/fingering)
        idx_start = int(0.05 * len(time_steps))
        idx_end = np.argmax(amp_history > (0.10 * params["NY"]))
        if idx_end <= idx_start:
            idx_end = len(time_steps) - 1  # Caso a perturbação não cresça tanto

        t_linear = time_steps[idx_start:idx_end]
        amp_linear = amp_history[idx_start:idx_end]

        if len(t_linear) < 10:
            print(f"[{params['id_caso']}] Falha: Janela linear insuficiente.")
            continue

        ln_amp = np.log(amp_linear + 1e-12)

        # Regressão Linear: ln(A) = omega * t + C
        coeffs = np.polyfit(t_linear, ln_amp, 1)
        omega_num = coeffs[0]

        erro_relativo = abs(omega_num - omega_theo) / abs(omega_theo) * 100.0 if omega_theo != 0 else 0.0

        print(f"\nCaso: {params['id_caso']}")
        print(f"Omega Teórico (Eq. 26) : {omega_theo:.6e}")
        print(f"Omega Numérico (LBM)   : {omega_num:.6e}")
        print(f"Erro Relativo          : {erro_relativo:.2f}%")

        # Plotagem Acadêmica
        plt.figure(figsize=(7, 5))
        plt.plot(t_linear, ln_amp, 'ko', markersize=3, label='Dados LBM')
        plt.plot(t_linear, np.polyval(coeffs, t_linear), 'r-', linewidth=2,
                 label=f'Regressão ($\\omega_{{num}}$ = {omega_num:.2e})')

        # Plot da inclinação teórica para comparação visual
        ln_amp_theo = omega_theo * t_linear + (ln_amp[0] - omega_theo * t_linear[0])
        plt.plot(t_linear, ln_amp_theo, 'b--', linewidth=2,
                 label=f'Teoria ($\\omega_{{theo}}$ = {omega_theo:.2e})')

        plt.title(f"Análise de Estabilidade Linear - {params['id_caso']}")
        plt.xlabel("Tempo (Iterações)")
        plt.ylabel(r"$\ln(A(t))$")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(folder, "LSA_Validation.png"), dpi=300)
        plt.close()


if __name__ == '__main__':
    process_lsa()