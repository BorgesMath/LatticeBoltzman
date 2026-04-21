import numpy as np
import matplotlib.pyplot as plt


def dispersion_magnetic(alpha, M, Da, Ca, Bo, Cam, Lam, H0n, H0t):
    """
    Relação de dispersão magnética decomposta em suas 4 parcelas fundamentais.
    """
    fator_mob = Da / (Ca * (1 + M))

    viscoso = alpha * (M - 1) / (M + 1)
    gravidade = fator_mob * alpha * Bo
    capilaridade = -fator_mob * (alpha ** 3)
    # Termo magnético: instabilizante para H0n, estabilizante para H0t
    magnetico = fator_mob * Cam * Lam * (alpha ** 2) * (H0n ** 2 - H0t ** 2)

    total = viscoso + gravidade + capilaridade + magnetico
    return viscoso, gravidade, capilaridade, magnetico, total


# =============================================================================
# PARÂMETROS BASE GERAIS
# =============================================================================
alpha_range = np.linspace(0, 6, 1000)

M_fixo = 2.0  # Regime viscoso instável
Da_base = 1.0
Ca_base = 1.0
Bo_base = 1.0
Cam_base = 1.0
Lam_base = 1.0

# Estilos monocromáticos padrão
line_styles = ['-', '--', '-.', ':']

# =============================================================================
# GRÁFICO 1: VARIAÇÃO DO CAMPO NORMAL (H0n)
# =============================================================================
fig1, ax1 = plt.subplots(figsize=(8, 5))
H0n_values = [0.0, 0.8, 1.2, 1.5]
H0t_fixo_g1 = 0.0

max_zeta_g1 = 0

for val, l_style in zip(H0n_values, line_styles):
    _, _, _, _, zeta = dispersion_magnetic(
        alpha_range, M_fixo, Da_base, Ca_base, Bo_base, Cam_base, Lam_base, val, H0t_fixo_g1
    )
    max_zeta_g1 = max(max_zeta_g1, np.max(zeta))

    ax1.plot(alpha_range, zeta, label=rf'$\tilde{{H}}_{{0n}} = {val}$',
             color='black', linestyle=l_style, linewidth=1.5)

    # Marcação da raiz (cut-off)
    idx_cross = np.where(np.diff(np.sign(zeta)))[0]
    for idx in idx_cross:
        if alpha_range[idx] > 0.1:
            ax1.plot(alpha_range[idx], 0, marker='o', markersize=6,
                     markerfacecolor='white', markeredgecolor='black', zorder=5)

ax1.axhline(0, color='black', linestyle='-', linewidth=0.8)
ax1.set_ylim(-max_zeta_g1 * 1.2, max_zeta_g1 * 1.2)
ax1.set_xlim(0, 5)

ax1.set_xlabel(r"Dimensionless wavenumber, $\alpha^*$", fontsize=12)
ax1.set_ylabel(r"Total growth rate, $\zeta^*$", fontsize=12)
ax1.tick_params(axis='both', which='major', labelsize=10, direction='in', top=True, right=True)
ax1.legend(fontsize=10, loc='lower left', frameon=False)
fig1.tight_layout()
fig1.savefig('1_spectra_normal_field.png', dpi=300)

# =============================================================================
# GRÁFICO 2: VARIAÇÃO DO CAMPO TANGENCIAL (H0t)
# =============================================================================
fig2, ax2 = plt.subplots(figsize=(8, 5))
H0n_fixo_g2 = 0.0
H0t_values = [0.0, 0.8, 1.2, 1.5]

max_zeta_g2 = 0

for val, l_style in zip(H0t_values, line_styles):
    _, _, _, _, zeta = dispersion_magnetic(
        alpha_range, M_fixo, Da_base, Ca_base, Bo_base, Cam_base, Lam_base, H0n_fixo_g2, val
    )
    if val == 0.0:  # O caso zero é o pico máximo aqui
        max_zeta_g2 = np.max(zeta)

    ax2.plot(alpha_range, zeta, label=rf'$\tilde{{H}}_{{0t}} = {val}$',
             color='black', linestyle=l_style, linewidth=1.5)

    idx_cross = np.where(np.diff(np.sign(zeta)))[0]
    for idx in idx_cross:
        if alpha_range[idx] > 0.1:
            ax2.plot(alpha_range[idx], 0, marker='o', markersize=6,
                     markerfacecolor='white', markeredgecolor='black', zorder=5)

ax2.axhline(0, color='black', linestyle='-', linewidth=0.8)
ax2.set_ylim(-max_zeta_g2 * 1.2, max_zeta_g2 * 1.2)
ax2.set_xlim(0, 3)

ax2.set_xlabel(r"Dimensionless wavenumber, $\alpha^*$", fontsize=12)
ax2.set_ylabel(r"Total growth rate, $\zeta^*$", fontsize=12)
ax2.tick_params(axis='both', which='major', labelsize=10, direction='in', top=True, right=True)
ax2.legend(fontsize=10, loc='upper right', frameon=False)  # Legenda no topo pois curvas caem
fig2.tight_layout()
fig2.savefig('2_spectra_tangential_field.png', dpi=300)

# =============================================================================
# GRÁFICO 3: DECOMPOSIÇÃO DAS COMPONENTES FÍSICAS (H0n > H0t)
# =============================================================================
fig3, ax3 = plt.subplots(figsize=(8, 6))

H0n_comp = 1.5
H0t_comp = 0.0

visc, grav, capil, mag, zeta_tot = dispersion_magnetic(
    alpha_range, M_fixo, Da_base, Ca_base, Bo_base, Cam_base, Lam_base, H0n_comp, H0t_comp
)

lbl_visc = r'Viscous: $\alpha^* \frac{M-1}{M+1}$'
lbl_grav = r'Gravity: $\frac{Da}{Ca(1+M)} \alpha^* Bo$'
lbl_mag = r'Magnetic: $\frac{Da}{Ca(1+M)} Ca_m \Lambda_m {\alpha^*}^2 (\tilde{H}_{0n}^2 - \tilde{H}_{0t}^2)$'
lbl_cap = r'Capillarity: $-\frac{Da}{Ca(1+M)} {\alpha^*}^3$'
lbl_tot = r'Total Growth Rate ($\zeta^*$)'

ax3.plot(alpha_range, visc, label=lbl_visc, color='black', linestyle='--', linewidth=1.5)
ax3.plot(alpha_range, grav, label=lbl_grav, color='gray', linestyle='-.', linewidth=1.5)
ax3.plot(alpha_range, mag, label=lbl_mag, color='gray', linestyle=(0, (3, 1, 1, 1)), linewidth=1.5)
ax3.plot(alpha_range, capil, label=lbl_cap, color='black', linestyle=':', linewidth=1.5)
ax3.plot(alpha_range, zeta_tot, label=lbl_tot, color='black', linestyle='-', linewidth=2.5)

ax3.axhline(0, color='black', linestyle='-', linewidth=0.8)

ax3.set_xlabel(r"Dimensionless wavenumber, $\alpha^*$", fontsize=12)
ax3.set_ylabel(r"Dimensionless growth rate components", fontsize=12)
ax3.tick_params(axis='both', which='major', labelsize=10, direction='in', top=True, right=True)

ax3.set_xlim(0, 4.5)
ax3.set_ylim(-10, 10)
ax3.legend(fontsize=10, loc='lower left', frameon=False)

fig3.tight_layout()
fig3.savefig('3_magnetic_components.png', dpi=300)

plt.show()