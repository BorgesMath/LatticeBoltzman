import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# --- FUNÇÃO DE FÍSICA ---
def get_max_growth_rate(M, Da, Ca, Bo, alpha_scan):
    """
    Para um conjunto fixo de parâmetros (M, Ca, Bo), varre o espectro de alpha
    e encontra a taxa de crescimento máxima (zeta_max).
    Se zeta_max < 0, o sistema é estável.
    """
    # Termo Viscoso
    termo_viscoso = alpha * (M - 1) / (M + 1)

    # Termo Potencial (Capilaridade + Gravidade + Darcy)
    # Proteção contra divisão por zero se Ca for muito pequeno
    Ca = np.maximum(Ca, 1e-9)
    fator = Da / (Ca * (1 + M))
    termo_potencial = fator * alpha_scan * (Bo - alpha_scan ** 2)

    zeta = termo_viscoso + termo_potencial

    # Retorna o máximo valor encontrado nesse espectro
    return np.max(zeta)


# --- CONFIGURAÇÃO DA VARREDURA ---
alpha_scan = np.linspace(0, 50, 500)  # Espectro de ondas para buscar o pico
Da_base = 1e-4

# Resolução do Mapa (Grid)
res = 100

# ==============================================================================
# MAPA 1: Capilaridade (Ca) vs. Viscosidade (M)
# Fixando Bo = 0 (Sem gravidade)
# ==============================================================================

# Eixos
M_vals = np.linspace(0, 2.0, res)  # 0 a 2 (Cruza M=1, ponto crítico)
Ca_vals = np.logspace(-5, -2, res)  # 1e-5 a 1e-2 (Escala Log)

# Criar Grid
M_grid, Ca_grid = np.meshgrid(M_vals, Ca_vals)
Z_map1 = np.zeros_like(M_grid)

# Loop de Cálculo (Pixel a Pixel)
print("Calculando Mapa 1 (Ca vs M)...")
for i in range(res):
    for j in range(res):
        Z_map1[i, j] = get_max_growth_rate(M_grid[i, j], Da_base, Ca_grid[i, j], 0.0, alpha_scan)

# ==============================================================================
# MAPA 2: Capilaridade (Ca) vs. Gravidade (Bo)
# Fixando M = 0.5 (Instável viscosamente)
# ==============================================================================

# Eixos
Bo_vals = np.linspace(-5, 5, res)  # Gravidade estabilizante < 0 < Instabilizante
Ca_vals_2 = np.logspace(-5, -2, res)

# Criar Grid
Bo_grid, Ca_grid_2 = np.meshgrid(Bo_vals, Ca_vals_2)
Z_map2 = np.zeros_like(Bo_grid)

# Loop de Cálculo
print("Calculando Mapa 2 (Ca vs Bo)...")
for i in range(res):
    for j in range(res):
        Z_map2[i, j] = get_max_growth_rate(0.5, Da_base, Ca_grid_2[i, j], Bo_grid[i, j], alpha_scan)

# ==============================================================================
# PLOTAGEM
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# --- PLOT 1 ---
# Usamos um colormap divergente (bwr ou RdBu) centrado em zero
# Níveis de contorno focados na transição
levels = np.linspace(-0.5, 2.0, 50)
norm = mcolors.TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=2.0)

cf1 = ax1.contourf(M_grid, Ca_grid, Z_map1, levels=levels, cmap='RdBu_r', norm=norm, extend='both')
cbar1 = fig.colorbar(cf1, ax=ax1)
cbar1.set_label(r'Taxa de Crescimento Máx ($\zeta^*_{max}$)', fontsize=12)

# Linha de Estabilidade Neutra (Zeta = 0)
cs1 = ax1.contour(M_grid, Ca_grid, Z_map1, levels=[0], colors='black', linewidths=2)
ax1.clabel(cs1, fmt='Neutro', fontsize=10)

ax1.set_yscale('log')
ax1.set_title(r"Mapa de Estabilidade: $Ca$ vs $M$ ($Bo=0$)", fontsize=14)
ax1.set_xlabel(r"Razão de Viscosidade ($M = \mu_2/\mu_1$)", fontsize=12)
ax1.set_ylabel(r"Número de Capilaridade ($Ca$)", fontsize=12)
ax1.axvline(1, color='green', linestyle='--', alpha=0.5, label='M=1 (Teórico)')
ax1.text(0.2, 1e-4, "INSTÁVEL", color='red', fontweight='bold', ha='center', fontsize=14,
         bbox=dict(facecolor='white', alpha=0.7))
ax1.text(1.5, 1e-4, "ESTÁVEL", color='blue', fontweight='bold', ha='center', fontsize=14,
         bbox=dict(facecolor='white', alpha=0.7))

# --- PLOT 2 ---
cf2 = ax2.contourf(Bo_grid, Ca_grid_2, Z_map2, levels=levels, cmap='RdBu_r', norm=norm, extend='both')
cbar2 = fig.colorbar(cf2, ax=ax2)
cbar2.set_label(r'Taxa de Crescimento Máx ($\zeta^*_{max}$)', fontsize=12)

# Linha de Estabilidade Neutra
cs2 = ax2.contour(Bo_grid, Ca_grid_2, Z_map2, levels=[0], colors='black', linewidths=2)
ax2.clabel(cs2, fmt='Neutro', fontsize=10)

ax2.set_yscale('log')
ax2.set_title(r"Mapa de Estabilidade: $Ca$ vs $Bo$ ($M=0.5$)", fontsize=14)
ax2.set_xlabel(r"Número de Bond ($Bo$ - Gravidade)", fontsize=12)
ax2.set_ylabel(r"Número de Capilaridade ($Ca$)", fontsize=12)
ax2.text(3, 1e-4, "GRAV. \nINSTÁVEL", color='darkred', ha='center', fontsize=10,
         bbox=dict(facecolor='white', alpha=0.7))
ax2.text(-3, 1e-4, "GRAV. \nESTABILIZANTE", color='darkblue', ha='center', fontsize=10,
         bbox=dict(facecolor='white', alpha=0.7))

plt.suptitle("Mapas de Estabilidade Linear (Azul = Estável, Vermelho = Instável)", fontsize=16, y=0.98)
plt.tight_layout()

# Salvar
plt.savefig("mapas_estabilidade_contour.png", dpi=300)
print("Mapas gerados e salvos em 'mapas_estabilidade_contour.png'")
plt.show()