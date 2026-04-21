import numpy as np
import matplotlib.pyplot as plt


def components_dispersion(alpha, M, Da, Ca, Bo):
    """
    Decomposes the dimensionless dispersion relation into its 3 physical phenomena:
    Viscous Forcing, Gravity, and Capillarity.
    """
    # 1. Viscous Forcing
    viscous = alpha * (M - 1) / (M + 1)

    # Common mobility factor for potential terms
    mobility = Da / (Ca * (1 + M))

    # 2. Gravity (Bond)
    gravity = mobility * alpha * Bo

    # 3. Capillarity (Surface Tension)
    capillarity = -mobility * (alpha ** 3)

    # Total Growth Rate
    total = viscous + gravity + capillarity

    return viscous, gravity, capillarity, total


# --- Parameters Configuration ---
alpha_range = np.linspace(0, 3, 500)

M_val = 2.0
Da_base = 1.0
Ca_base = 1.0
Bo_base = 1.0

visc, grav, capil, zeta_total = components_dispersion(alpha_range, M_val, Da_base, Ca_base, Bo_base)

# --- Plot Configuration ---
fig, ax = plt.subplots(figsize=(8, 6))

# LaTeX Labels with equations
label_visc = r'Viscous Forcing: $\alpha^* \frac{M-1}{M+1}$'
label_grav = r'Gravity: $\frac{Da}{Ca(1+M)} \alpha^* Bo$'
label_capil = r'Capillarity: $-\frac{Da}{Ca(1+M)} {\alpha^*}^3$'
label_total = r'Total Growth Rate ($\zeta^*$)'

# Plotting components
ax.plot(alpha_range, visc, label=label_visc, color='black', linestyle='--', linewidth=1.5)
ax.plot(alpha_range, grav, label=label_grav, color='black', linestyle='-.', linewidth=1.5)
ax.plot(alpha_range, capil, label=label_capil, color='black', linestyle=':', linewidth=1.5)

# Total resultant curve
ax.plot(alpha_range, zeta_total, label=label_total, color='black', linestyle='-', linewidth=2.5)

# --- Academic Formatting ---
ax.axhline(0, color='black', linestyle='-', linewidth=0.8)

ax.set_xlabel(r"Dimensionless wavenumber, $\alpha^*$", fontsize=12)
ax.set_ylabel(r"Dimensionless growth rate, $\zeta^*$", fontsize=12)

ax.tick_params(axis='both', which='major', labelsize=10, direction='in', top=True, right=True)

ax.set_xlim(0, 3)
ax.set_ylim(-5, 4)

ax.legend(fontsize=11, loc='lower left', frameon=False)

plt.tight_layout()
plt.show()