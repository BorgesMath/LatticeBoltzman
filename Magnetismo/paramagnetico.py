# Magnetismo/paramagnetico.py
import numpy as np
from numba import njit


@njit
def compute_paramagnetic_field_analytical(psi, amplitude_lambda, nx, ny, k, H0t, H0n1, mu1, mu2, x_interface):
    """
    Constrói o campo potencial magnético (psi) ANALITICAMENTE.
    VERSÃO ROTACIONADA PARA INTERFACE VERTICAL (LBM).

    Geometria:
    - Interface média em x = x_interface (ex: 80)
    - Normal à interface: Eixo X
    - Tangencial à interface: Eixo Y
    - Perturbação varia com cos(k*y)
    """

    # Continuidade do campo B normal (agora componente X)
    # H0n1 é o campo em X no fluido 1 (Invasor/Esquerda)
    H0n2 = (mu1 / mu2) * H0n1

    denom = mu1 + mu2

    # Constantes baseadas na teoria (Eq. 97 do PDF adaptada para geometria X)
    term_Hn = -mu2 * (H0n2 - H0n1)
    C1 = (amplitude_lambda / denom) * term_Hn
    C2 = C1 + amplitude_lambda * (H0n2 - H0n1)

    for y_idx in range(ny):
        # Coordenada Y (Tangencial)
        # A onda varia em Y: cos(k * y)
        theta = np.cos(k * y_idx)

        # Potencial Base Tangencial (H0t * y)
        psi_tangencial_base = -H0t * y_idx

        for x_idx in range(nx):
            # Coordenada X relativa à interface (Normal)
            x_phys = x_idx - x_interface

            if x_phys < 0:  # Meio 1 (Esquerda / Invasor)
                # Base: -H0n1 * x - H0t * y
                psi_base = -H0n1 * x_phys + psi_tangencial_base

                # Perturbação: decai para x -> -infinito (exponencial positiva de x negativo)
                psi_pert = C1 * np.exp(k * x_phys) * theta

                psi[y_idx, x_idx] = psi_base + psi_pert

            else:  # Meio 2 (Direita / Residente)
                # Base
                psi_base = -H0n2 * x_phys + psi_tangencial_base

                # Perturbação: decai para x -> +infinito (exponencial negativa)
                psi_pert = C2 * np.exp(-k * x_phys) * theta

                psi[y_idx, x_idx] = psi_base + psi_pert

    return psi