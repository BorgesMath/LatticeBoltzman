import numpy as np
import matplotlib.pyplot as plt


def dispersion_relation(alpha, M, Da, Ca, Bo):
    # Termo A: Instabilidade Viscosa (Saffman-Taylor clássico)
    # Se M < 1 (menos viscoso empurra mais viscoso), este termo é positivo
    termo_viscoso = alpha * (1 - M) / (1 + M)

    # Termo B: Estabilização por Tensão Superficial (termo cúbico em alpha)
    fator_mobilidade = Da / (Ca * (1 + M))
    termo_potencial = fator_mobilidade * alpha * (Bo - alpha ** 2)

    zeta = termo_viscoso + termo_potencial
    return zeta


# --- Configuração dos Parâmetros Físicos ---
# REDUZIDO: Focando em ondas longas (0 a 5) para ver o início da instabilidade
alpha_range = np.linspace(0, 100, 500)

# Parâmetros Base
Da_base = 1e-4
Ca_base = 1e-3
Bo_base = 0.0

# Configuração do Plot
fig, ax = plt.subplots(figsize=(10, 6))

# Casos de Estudo: Apenas casos instáveis (M < 1) para visualizar o crescimento
M_values = [0.0, 0.2, 0.5, 0.8]  # M=0 é o caso mais instável (viscosidade desprezível empurrando óleo)

max_zeta_global = 0  # Para ajustar o zoom vertical automaticamente

for M_val in M_values:
    zeta = dispersion_relation(alpha_range, M_val, Da_base, Ca_base, Bo_base)

    # Atualiza o máximo global para o zoom
    if np.max(zeta) > max_zeta_global:
        max_zeta_global = np.max(zeta)

    # Plotar a curva
    ax.plot(alpha_range, zeta, label=f'M={M_val} (Instável)', linewidth=2.5)

    # Marcar o ponto de corte (onde a instabilidade morre devido à tensão superficial)
    # Procura onde cruza zero vindo do positivo
    idx_cross = np.where(np.diff(np.sign(zeta)))[0]
    for idx in idx_cross:
        if zeta[idx] > 0 and alpha_range[idx] > 0.1:  # Ignora a origem
            ax.scatter(alpha_range[idx], 0, color='black', marker='|', s=100)

# --- AJUSTES VISUAIS DE ZOOM ---

# Linha Neutra (Zero) bem destacada
ax.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)

# Preenchimento da área instável (apenas decorativo para o caso mais crítico)
zeta_critical = dispersion_relation(alpha_range, 0.0, Da_base, Ca_base, Bo_base)
ax.fill_between(alpha_range, 0, zeta_critical, where=(zeta_critical > 0),
                color='red', alpha=0.05, label='Região Instável')

# ZOOM: Limitar o Eixo Y para mostrar apenas o "morrinho" positivo
# Vai de um pouco negativo (-0.02) até 10% acima do pico máximo
ax.set_ylim(-10, max_zeta_global * 1.5)

# Limitar X para a região de interesse
ax.set_xlim(0, 8)

# Decoração
ax.set_title(r"Zoom na Região de Instabilidade ($\zeta^* > 0$)", fontsize=16)
ax.set_xlabel(r"Número de Onda Adimensional ($\alpha^*$)", fontsize=14)
ax.set_ylabel(r"Taxa de Crescimento ($\zeta^*$)", fontsize=14)
ax.grid(True, which='both', linestyle='--', alpha=0.4)
ax.legend(fontsize=12, loc='upper right')

plt.tight_layout()
plt.show()

