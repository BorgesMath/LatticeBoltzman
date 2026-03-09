import numpy as np
import matplotlib.pyplot as plt


def dispersion_relation(alpha, M, Da, Ca, Bo):
    # Termo A: Instabilidade Viscosa
    termo_viscoso = alpha * (1 - M) / (1 + M)

    # Termo B: Potencial (Darcy + Capilaridade + Gravidade)
    fator_mobilidade = Da / (Ca * (1 + M))
    termo_potencial = fator_mobilidade * alpha * (Bo - alpha ** 2)

    return termo_viscoso + termo_potencial


# --- Configuração Geral ---
alpha_calc = np.linspace(0, 20, 1000)
Da_base = 1e-4

# Criar figura
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# ==============================================================================
# GRÁFICO 1: VARIAÇÃO DE VISCOSIDADE (M)
# ==============================================================================
Ca_fix = 1e-3
Bo_fix = 0.0
M_values = [0.0, 0.2, 0.5, 0.8]

for M in M_values:
    zeta = dispersion_relation(alpha_calc, M, Da_base, Ca_fix, Bo_fix)
    ax1.plot(alpha_calc, zeta, label=f'M={M}', linewidth=2)

ax1.set_title(r"Efeito da Viscosidade ($M$)", fontsize=14)
ax1.set_xlabel(r"$\alpha^*$", fontsize=12)
ax1.set_ylabel(r"$\zeta^*$", fontsize=12)
ax1.set_xlim(0, 10)
# Zoom dinâmico
ymax_M = np.max(dispersion_relation(alpha_calc, 0.0, Da_base, Ca_fix, Bo_fix))
ax1.set_ylim(-1, ymax_M * 1.2)
ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# ==============================================================================
# GRÁFICO 2: VARIAÇÃO DE CAPILARIDADE (Ca)
# ==============================================================================
M_fix = 0.1
Bo_fix = 0.0
Ca_values = [1e-4, 5e-4, 1e-3, 5e-3]

for Ca in Ca_values:
    zeta = dispersion_relation(alpha_calc, M_fix, Da_base, Ca, Bo_fix)
    ax2.plot(alpha_calc, zeta, label=f'Ca={Ca:.0e}', linewidth=2)

ax2.set_title(r"Efeito da Capilaridade ($Ca$)", fontsize=14)
ax2.set_xlabel(r"$\alpha^*$", fontsize=12)
ax2.set_xlim(0, 15)
ymax_Ca = np.max(dispersion_relation(alpha_calc, M_fix, Da_base, 5e-3, Bo_fix))
ax2.set_ylim(-1, ymax_Ca * 1.2)
ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right')

# ==============================================================================
# GRÁFICO 3: VARIAÇÃO DE GRAVIDADE (Bo)
# ==============================================================================
M_fix = 0.1
Ca_fix = 1e-3
Bo_values = [-2.0, 0.0, 2.0, 5.0]

for Bo in Bo_values:
    zeta = dispersion_relation(alpha_calc, M_fix, Da_base, Ca_fix, Bo)
    label = f'Bo={Bo}'
    ax3.plot(alpha_calc, zeta, label=label, linewidth=2)

ax3.set_title(r"Efeito da Gravidade ($Bo$)", fontsize=14)
ax3.set_xlabel(r"$\alpha^*$", fontsize=12)
ax3.set_xlim(0, 10)
ymax_Bo = np.max(dispersion_relation(alpha_calc, M_fix, Da_base, Ca_fix, 5.0))
ax3.set_ylim(-2, ymax_Bo * 1.2)
ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper right')

# --- FINALIZAÇÃO E SALVAMENTO ---
plt.suptitle("Análise de Estabilidade Linear: Saffman-Taylor Generalizado", fontsize=16, y=0.98)
plt.tight_layout()

# Salvar o gráfico
nome_arquivo = "instabilidade_saffman_taylor_comparativo.png"
plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
print(f"Gráfico salvo com sucesso em: {nome_arquivo}")

plt.show()