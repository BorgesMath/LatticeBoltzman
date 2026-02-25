# lsa.py
import numpy as np
import matplotlib.pyplot as plt
import os
from config import TAU_IN, TAU_OUT, SIGMA, U_INLET, H0, CHI_MAX, NY


def compute_dimensionless_numbers():
    """
    Ponte de transferência: Converte parâmetros LBM para números adimensionais.
    Nota: Ajuste os fatores de escala espacial/permeabilidade conforme as
    definições exatas da sua tese geométrica.
    """
    # Viscosidades Cinemáticas LBM (nu = (tau - 0.5) / 3)
    nu_in = (TAU_IN - 0.5) / 3.0
    nu_out = (TAU_OUT - 0.5) / 3.0

    # Razão de mobilidade (Viscosidade relativa)
    lamb = nu_in / nu_out

    # Contraste Magnético
    Lambda = CHI_MAX

    # Estimativas Adimensionais (Requerem calibração com o comprimento de referência L)
    # Aqui utilizamos constantes proporcionais aos tensores do config.py
    Da = 0.1  # Permeabilidade adimensional (fixado conforme seu script original)
    Ca = (nu_in * U_INLET) / SIGMA  # Número Capilar (Viscosas / Capilares)

    # Bond Magnético (Forças Magnéticas / Capilares)
    # Proporcional a H0^2 * Chi / Sigma
    Bom = (CHI_MAX * H0 ** 2) / SIGMA

    return Da, Ca, lamb, Lambda, Bom


def taxa_crescimento_normal(k, Bom, Da, Ca, lamb, Lambda):
    """
    Equação de dispersão para campo normal (theta = 0, fator angular = 1).
    s = [Da * k / (Ca * (1 + lamb))] * [ -k^2 + Bom * k * Lambda ]
    """
    pre_fator = (Da * k) / (Ca * (1.0 + lamb))
    termo_colchetes = -(k ** 2) + (Bom * k * Lambda)
    return pre_fator * termo_colchetes


def analyze_stability(mode_m, output_dir):
    """
    Avalia se o modo m inserido na inicialização crescerá ou decairá.
    """
    Da, Ca, lamb, Lambda, Bom = compute_dimensionless_numbers()

    # 1. Determinação do Número de Onda da Simulação
    # k = 2 * pi * m / L (Onde L é a largura do domínio Y)
    k_sim = 2.0 * np.pi * mode_m / NY

    # 2. Avaliação da Taxa de Crescimento para o modo específico
    s_sim = taxa_crescimento_normal(k_sim, Bom, Da, Ca, lamb, Lambda)

    # 3. Geração do Espectro Contínuo para o Plot
    k_array = np.linspace(0, max(5.0, k_sim * 2), 500)
    s_array = taxa_crescimento_normal(k_array, Bom, Da, Ca, lamb, Lambda)

    # --- Plot e Exportação ---
    plt.figure(figsize=(10, 6))
    plt.plot(k_array, s_array, 'k-', linewidth=2, label='Curva de Dispersão LSA')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)

    # Destacar o ponto exato que será simulado
    plt.plot(k_sim, s_sim, 'ro', markersize=8, label=f'Modo Simulado (m={mode_m})')

    plt.title(f"Análise de Estabilidade Linear - LBM config.py\n$Bo_m \approx {Bom:.2f}, Ca \approx {Ca:.2f}$")
    plt.xlabel(r'Número de Onda ($k$)')
    plt.ylabel(r'Taxa de Crescimento ($s$)')

    # Determinação do regime
    regime = "INSTÁVEL (Crescimento de Dedos)" if s_sim > 0 else "ESTÁVEL (Atenuação da Perturbação)"
    plt.text(0.05, 0.95, f"Regime Previsto: {regime}\n$s = {s_sim:.4e}$",
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"analise_lsa_modo_{mode_m}.png"), dpi=150)
    plt.close()

    # Retorno no console para decisão em tempo de execução
    print(f"--- Diagnóstico Analítico (LSA) ---")
    print(f"Modo m={mode_m} | k={k_sim:.4f} | Taxa s={s_sim:.4e}")
    print(f"Previsão Teórica: Interface {regime}")
    print(f"-----------------------------------\n")

    return s_sim > 0  # Retorna booleano indicando instabilidade