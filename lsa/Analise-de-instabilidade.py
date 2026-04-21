import numpy as np
import matplotlib.pyplot as plt


def dispersion_relation(alpha, M, Da, Ca, Bo):
    termo_viscoso = alpha * (M - 1) / (M + 1)
    fator_mobilidade = Da / (Ca * (1 + M))
    termo_potencial = fator_mobilidade * alpha * (Bo - alpha ** 2)

    return termo_viscoso + termo_potencial


# --- Configuração dos Parâmetros Físicos ---
alpha_range = np.linspace(0, 15, 1000)

Da_base = 1
Ca_base = 1
Bo_base = 1

# Figura com proporção otimizada para colunas de artigos
fig, ax = plt.subplots(figsize=(8, 5))

M_values = [1]
# Estilos de linha distintos para visualização em escala de cinza

line_styles = ['-']

max_zeta_global = 0

for M_val, l_style in zip(M_values, line_styles):
    zeta = dispersion_relation(alpha_range, M_val, Da_base, Ca_base, Bo_base)

    if np.max(zeta) > max_zeta_global:
        max_zeta_global = np.max(zeta)

    # Plotagem monocromática
    label = rf'$M={M_val}$, $Da={Da_base}$, $Ca={Ca_base}$, $Bo={Bo_base}$'
    ax.plot(alpha_range, zeta, label=label, color='black',
            linestyle=l_style, linewidth=1.5)

    # Marcador do cut-off wavenumber (raiz)
    idx_cross = np.where(np.diff(np.sign(zeta)))[0]
    for idx in idx_cross:
        if alpha_range[idx] > 0.1:
            ax.plot(alpha_range[idx], 0, marker='o', markersize=6,
                    markerfacecolor='white', markeredgecolor='black', zorder=5)

# --- AJUSTES VISUAIS ACADÊMICOS ---

# Linha de base neutra
ax.axhline(0, color='black', linestyle='-', linewidth=0.8)

# Simetria do eixo Y
zoom_factor = 1.5
ax.set_ylim(-4, max_zeta_global * zoom_factor)
ax.set_xlim(0, 5)

# Rótulos (Omitiu-se o título geral, assumindo o uso de caption no documento de texto)
ax.set_xlabel(r"Dimensionless wavenumber, $\alpha^*$", fontsize=12)
ax.set_ylabel(r"Dimensionless growth rate, $\zeta^*$", fontsize=12)

# Configuração de ticks estilo publicação (Physical Review, JFM, etc.)
ax.tick_params(axis='both', which='major', labelsize=10, direction='in',
               top=True, right=True)

# Legenda minimalista
ax.legend(fontsize=10, loc='upper right', frameon=False)

plt.tight_layout()
plt.show()