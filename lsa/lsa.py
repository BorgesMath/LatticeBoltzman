# lsa/lsa.py
import numpy as np
import matplotlib.pyplot as plt
import os
from config.config import TAU_IN, TAU_OUT, SIGMA, U_INLET, H0, CHI_MAX, NY, K_0, H_ANGLE


def analyze_stability(mode_m, output_dir):
    """
    Realiza a Análise de Estabilidade Linear (LSA) rigorosamente em Lattice Units (LU).
    Baseado na formulação analítica que acopla Saffman-Taylor com Paramagnetismo.
    """

    # 1. Propriedades Hidrodinâmicas (Viscosidade Dinâmica = Cinemática pois rho=1)
    eta_in = (TAU_IN - 0.5) / 3.0
    eta_out = (TAU_OUT - 0.5) / 3.0

    # 2. Propriedades Magnéticas Vetoriais
    angle_rad = np.radians(H_ANGLE)
    H0n = H0 * np.cos(angle_rad)  # Componente Normal
    H0t = H0 * np.sin(angle_rad)  # Componente Tangencial

    mu0 = 1.0
    mu1 = mu0 * (1.0 + CHI_MAX)
    mu2 = mu0
    fator_mu = (mu1 - mu2) / (mu1 + mu2)

    # 3. Número de Onda (k) Dimensional (LU)
    k_sim = 2.0 * np.pi * mode_m / NY

    def taxa_crescimento_dimensional(k):
        # A. Força Motriz Viscosa (Desestabilizante)
        t_visc = ((eta_out - eta_in) / K_0) * U_INLET

        # B. Força Restauradora Capilar (Estabilizante)
        t_cap = - (k ** 2) * SIGMA

        # C. Força Magnética Direcional
        t_mag = mu0 * CHI_MAX * k * fator_mu * ((H0n ** 2) - (H0t ** 2))

        # Mobilidade Global
        pre_fator = (K_0 * k) / (eta_in + eta_out)

        return pre_fator * (t_visc + t_cap + t_mag)

    # Cálculo da taxa para o modo simulado
    s_sim = taxa_crescimento_dimensional(k_sim)

    # Espectro Contínuo
    k_array = np.linspace(0.001, max(0.5, k_sim * 3), 1000)
    s_array = taxa_crescimento_dimensional(k_array)

    # --- Plotagem Acadêmica ---
    plt.figure(figsize=(10, 6))
    plt.plot(k_array, s_array, 'k-', linewidth=2, label='Curva de Dispersão LSA (Teórica)')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.plot(k_sim, s_sim, 'ro', markersize=8, label=f'Modo Simulado ($m={mode_m}$)')

    # Anotação das forças
    t_v = ((eta_out - eta_in) / K_0) * U_INLET
    t_c = - (k_sim ** 2) * SIGMA
    t_m = mu0 * CHI_MAX * k_sim * fator_mu * ((H0n ** 2) - (H0t ** 2))
    anotacao_forcas = (
        f"Contribuições de Força:\n"
        f"Viscosa: {t_v:.2e}\n"
        f"Capilar: {t_c:.2e}\n"
        f"Magnética: {t_m:.2e}"
    )
    plt.text(0.95, 0.95, anotacao_forcas, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.title(fr"Análise de Estabilidade Linear Exata (Lattice Units)")
    plt.xlabel(r'Número de Onda $k$ ($pixel^{-1}$)')
    plt.ylabel(r'Taxa de Crescimento $Re(s)$ ($ts^{-1}$)')

    regime = "INSTÁVEL" if s_sim > 0 else "ESTÁVEL"
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"analise_lsa_modo_{mode_m}.png"), dpi=150)
    plt.close()

    print(f"--- Diagnóstico Analítico (LSA) ---")
    print(f"Taxa Teórica de Crescimento (s): {s_sim:.6e} ts^-1")
    print(f"Regime Previsto: {regime}")
    print(f"-----------------------------------")

    return s_sim