import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Importações estritas do modelo físico
from config.config import (NY, NX, TAU_IN, TAU_OUT, U_INLET, K_0, SIGMA,
                           W_LBM, CX, CY, INTERFACE_WIDTH)
from lbm.lbm import lbm_step
from Multifasico.cahn_hilliard import cahn_hilliard_substep

# ==============================================================================
# 1. FORMULAÇÃO TEÓRICA E DEFINIÇÃO DOS REGIMES
# ==============================================================================
nu_in = (TAU_IN - 0.5) / 3.0
nu_out = (TAU_OUT - 0.5) / 3.0
M_hydro = K_0 / (nu_in + nu_out)


def omega_teorico(k):
    """
    Taxa de crescimento linear analítica (Darcy-Brinkman acoplado a Saffman-Taylor).
    """
    forca_viscosa = ((nu_out - nu_in) / K_0) * U_INLET
    forca_capilar = - (k ** 2) * SIGMA
    return M_hydro * k * (forca_viscosa + forca_capilar)


# Cálculo do Ponto Crítico Marginal (\omega = 0)
k_critico = np.sqrt(((nu_out - nu_in) * U_INLET) / (K_0 * SIGMA))
modo_critico = k_critico * NY / (2.0 * np.pi)

print(f"--- PARÂMETROS ANALÍTICOS ---")
print(f"Modo Crítico de Corte (m_c): {modo_critico:.2f}")
print(f"Número de Onda Crítico (k_c): {k_critico:.6e} LU^-1")

# Seleção do Espectro: Modos antes de m_c (Instáveis) e após m_c (Estáveis)
m_instaveis = [2, 4, int(modo_critico * 0.5), int(modo_critico * 0.8)]
m_estaveis = [int(modo_critico * 1.2), int(modo_critico * 1.5), int(modo_critico * 2.0)]
modos_teste = sorted(list(set([m for m in m_instaveis + m_estaveis if m > 0])))


# ==============================================================================
# 2. MÉTRICA CONTÍNUA DA AMPLITUDE INTERFACIAL (\lambda)
# ==============================================================================
def extrair_amplitude_lambda(phi):
    """
    Localização sub-pixel da interface (phi = 0) para medição de \lambda.
    """
    ny_shape, nx_shape = phi.shape
    x_positions = []

    for y in range(ny_shape):
        crossings = np.where(np.diff(np.sign(phi[y, :])))[0]
        if len(crossings) > 0:
            x_idx = crossings[-1]
            phi0 = phi[y, x_idx]
            phi1 = phi[y, x_idx + 1]

            if np.abs(phi1 - phi0) > 1e-6:
                x_sub = x_idx - phi0 / (phi1 - phi0)
                x_positions.append(x_sub)

    if not x_positions: return 0.0
    return (np.max(x_positions) - np.min(x_positions)) / 2.0


# ==============================================================================
# 3. ROTINA DE INTEGRAÇÃO LBM POR MODO
# ==============================================================================
def simular_modo_lsa(m):
    print(f"Processando Modo m={m} | Regime: {'Instável' if m < modo_critico else 'Estável'}")

    f = np.zeros((NY, NX, 9), dtype=np.float64)
    phi = np.zeros((NY, NX), dtype=np.float64)
    psi = np.zeros((NY, NX), dtype=np.float64)  # Magnético nulo
    chi = np.zeros((NY, NX), dtype=np.float64)
    K_field = np.ones((NY, NX), dtype=np.float64) * K_0
    rho = np.ones((NY, NX), dtype=np.float64)

    # Inicialização da Perturbação
    lambda_0 = 1.5
    x_center = NX * 0.4
    k_onda = 2.0 * np.pi * m / NY

    for y in range(NY):
        pos_x = x_center + lambda_0 * np.sin(k_onda * y)
        for x in range(NX):
            phi[y, x] = -np.tanh((x - pos_x) / (INTERFACE_WIDTH / 2.0))

    # Equilíbrio de Fase Prévio
    u_zero = np.zeros((NY, NX), dtype=np.float64)
    for _ in range(50):
        phi = cahn_hilliard_substep(phi, u_zero, u_zero)

    u_x = np.ones((NY, NX), dtype=np.float64) * U_INLET
    u_y = np.zeros((NY, NX), dtype=np.float64)
    u_sq = u_x ** 2 + u_y ** 2

    # Inicialização das Populações de Momento
    for y in range(NY):
        for x in range(NX):
            for i in range(9):
                cu = CX[i] * u_x[y, x] + CY[i] * u_y[y, x]
                f[y, x, i] = W_LBM[i] * rho[y, x] * (1.0 + 3.0 * cu + 4.5 * cu ** 2 - 1.5 * u_sq[y, x])

    tempo_max = 1200  # Restrito à fase linear
    janela_amostragem = 10

    t_hist, lambda_hist = [], []

    for t in range(tempo_max):
        for _ in range(10):  # CH Substeps
            phi = cahn_hilliard_substep(phi, u_x, u_y)

        f, rho, u_x, u_y = lbm_step(f, phi, psi, rho, u_x, u_y, chi, K_field)

        if t % janela_amostragem == 0:
            amp = extrair_amplitude_lambda(phi)
            if amp > 1e-4:  # Filtro de dissipação total (evita erro no logaritmo)
                t_hist.append(t)
                lambda_hist.append(amp)

    return np.array(t_hist), np.array(lambda_hist)


# ==============================================================================
# 4. EXECUÇÃO E PROCESSAMENTO ESTATÍSTICO DAS TAXAS (\omega)
# ==============================================================================
k_numerico = []
omega_numerico = []

plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

for m in modos_teste:
    t_arr, amp_arr = simular_modo_lsa(m)

    if len(t_arr) < 5:
        continue

    # Isolamento do regime puramente linear (descarta inércia inicial e saturação tardia)
    corte_ini = int(len(t_arr) * 0.15)
    corte_fim = int(len(t_arr) * 0.85)

    t_lin = t_arr[corte_ini:corte_fim]
    amp_lin = amp_arr[corte_ini:corte_fim]

    # Regressão Logarítmica: ln(\lambda) = \omega * t + C
    coeficientes = np.polyfit(t_lin, np.log(amp_lin), 1)
    omega_extratado = coeficientes[0]

    k_onda = 2.0 * np.pi * m / NY
    k_numerico.append(k_onda)
    omega_numerico.append(omega_extratado)

    # Traçado das evoluções de lambda (Painel Secundário)
    ax2.plot(t_arr, amp_arr, label=rf'Modo {m} ($\omega_{{num}}$={omega_extratado:.2e})')

# ==============================================================================
# 5. DIAGRAMAÇÃO DO ESPECTRO DE DISPERSÃO
# ==============================================================================
k_continuo = np.linspace(0, max(k_numerico) * 1.2, 500)
omega_continuo = omega_teorico(k_continuo)

# Painel Principal: Curva de Dispersão LSA
ax1.plot(k_continuo, omega_continuo, 'k-', linewidth=2.5, label='Curva de Dispersão Teórica')
ax1.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Margem de Estabilidade Neutral')
ax1.plot(k_numerico, omega_numerico, 'bo', markersize=9, zorder=5, markeredgecolor='black',
         label='Extração Numérica (LBM)')

# Sombreamento dos Regimes
ax1.fill_between(k_continuo, 0, omega_continuo, where=(omega_continuo >= 0), color='red', alpha=0.1,
                 label='Regime Instável (Crescimento)')
ax1.fill_between(k_continuo, 0, omega_continuo, where=(omega_continuo < 0), color='blue', alpha=0.1,
                 label='Regime Estável (Dissipação)')

ax1.set_xlabel(r'Número de Onda $k$ ($LU^{-1}$)', fontsize=12)
ax1.set_ylabel(r'Taxa de Crescimento $\omega = Re(s)$ ($LU^{-1}$)', fontsize=12)
ax1.set_title('Validação Espectral Multimodo LSA', fontsize=14, weight='bold')
ax1.legend(loc='best')
ax1.grid(True, linestyle=':', alpha=0.7)

# Painel Secundário: Dinâmica de Amplitude
ax2.set_yscale('log')
ax2.set_xlabel('Tempo de Integração $t$ (passos)', fontsize=12)
ax2.set_ylabel(r'Amplitude Interfacial Sub-pixel $\lambda(t)$', fontsize=12)
ax2.set_title('Evolução Temporal da Perturbação', fontsize=14, weight='bold')
ax2.legend(loc='upper right', fontsize=8)
ax2.grid(True, which='both', linestyle=':', alpha=0.5)

plt.tight_layout()
caminho_grafico = os.path.join(os.getcwd(), 'analise_lsa_espectro_completo.png')
plt.savefig(caminho_grafico, dpi=300)
print(f"--- EXECUÇÃO CONCLUÍDA ---")
print(f"Diagrama de validação em malha contínua gerado: {caminho_grafico}")