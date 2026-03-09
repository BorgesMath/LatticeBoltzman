# Magnetismo/paramagnetico.py
import numpy as np
from numba import njit


@njit
def compute_paramagnetic_field_analytical(psi, x_grid, y_grid, amplitude_lambda, params):
    """
    Constrói o campo potencial magnético (psi) ANALITICAMENTE baseado nas
    derivações do PDF (Eqs. 30, 45, 93, 97).

    Isso substitui o solver de Poisson numérico por uma imposição direta da
    solução teórica linearizada.
    """
    nx = params['nx']
    ny = params['ny']
    k = params['k']  # Número de onda
    H0t = params['H0t']  # Campo Tangencial Base
    H0n1 = params['H0n1']  # Campo Normal Base (Meio 1)
    mu1 = params['mu1']
    mu2 = params['mu2']
    y_interface = params['y_interface']  # Posição média da interface (NY/2)

    # Continuidade do campo B normal no estado base (Eq. 24)
    # H0n2 = (mu1/mu2) * H0n1
    H0n2 = (mu1 / mu2) * H0n1

    # Cálculo das Constantes de Integração (Eq. 97 e 93)
    # Termo comum: i * H0t * (mu2 - mu1) ... note que na simulação real usamos cos/sin.
    # Assumindo fase real cos(kx):
    # O termo imaginário 'i' na análise espectral vira mudança de fase ou amplitude
    # dependendo da base. Para cos(kx), a parte real da resposta domina.

    # Simplificação: O PDF trata exp(ikx). A parte real é cos(kx).
    # Vamos implementar a amplitude real da perturbação no potencial.

    denom = mu1 + mu2

    # Eq. 97 adaptada para parte Real (considerando perturbação em cos(kx))
    # C1 = (lambda / (mu1+mu2)) * [ -mu2(H0n2 - H0n1) ]  <-- Termo associado a cos(kx)
    # O termo com 'i' no PDF (iH0t...) estaria associado a sin(kx).
    # Focaremos no termo principal que acopla com a deformação geométrica direta.

    term_Hn = -mu2 * (H0n2 - H0n1)
    C1 = (amplitude_lambda / denom) * term_Hn

    # Eq. 93
    C2 = C1 + amplitude_lambda * (H0n2 - H0n1)

    for y_idx in range(ny):
        # Coordenada y relativa à interface (y=0 no PDF)
        y_phys = y_idx - y_interface

        for x_idx in range(nx):
            # Coordenada x
            x_phys = x_idx

            # Termo oscilatório da perturbação (Eq. 45: theta = e^{ikx})
            # Usamos cos(k*x) pois a simulação é real
            theta = np.cos(k * x_phys)

            if y_phys < 0:  # Meio 1 (Inferior no PDF, y<0)
                # Eq. 30 (Base) + Eq. 45 (Perturbação)
                psi_base = -H0t * x_phys - H0n1 * y_phys
                psi_pert = C1 * np.exp(k * y_phys) * theta
                psi[y_idx, x_idx] = psi_base + psi_pert

            else:  # Meio 2 (Superior no PDF, y>0)
                # Eq. 30 (Base) + Eq. 45 (Perturbação)
                psi_base = -H0t * x_phys - H0n2 * y_phys
                psi_pert = C2 * np.exp(-k * y_phys) * theta
                psi[y_idx, x_idx] = psi_base + psi_pert

    return psi