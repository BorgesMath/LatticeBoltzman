# Magnetismo/paramagnetico.py
import numpy as np
from numba import njit


@njit
def compute_paramagnetic_field_analytical(psi, amplitude_lambda, nx, ny, k, H0t, H0n1, mu1, mu2, x_interface_mean):
    """
    Constrói o campo potencial magnético (psi) ANALITICAMENTE com REGULARIZAÇÃO SUAVE.

    Diferença chave para a versão anterior:
    Em vez de cortar a solução abruptamente com if/else (que cria derivada infinita),
    usamos funções sigmoidais (tanh) para misturar as soluções dos dois meios.
    Isso garante que a Força Magnética (derivada do potencial) seja sempre finita e estável.
    """

    # 1. Propriedades Magnéticas
    # H0n2 calculado pela continuidade de B (Eq. 24 do PDF)
    H0n2 = (mu1 / mu2) * H0n1
    denom = mu1 + mu2

    # Constantes da Perturbação (Eq. 97)
    # C2 - C1 = lambda * (H0n2 - H0n1)
    term_Hn = -mu2 * (H0n2 - H0n1)

    # Limitador de segurança para amplitude (evita números gigantes se a simulação instabilizar)
    safe_amp = max(-50.0, min(50.0, amplitude_lambda))

    C1 = (safe_amp / denom) * term_Hn
    C2 = C1 + safe_amp * (H0n2 - H0n1)

    # 2. Parâmetros de Suavização (Regularização)
    # w_smooth define a largura da transição física entre os meios.
    # 4.0 a 6.0 é ideal para evitar o "bico" sem perder a física da interface.
    w_smooth = 5.0
    inv_w = 1.0 / w_smooth

    # Pré-cálculo de médias e diferenças para o campo base suave
    H_avg = (H0n1 + H0n2) * 0.5
    H_diff = (H0n2 - H0n1) * 0.5

    for y_idx in range(ny):
        # Termo oscilatório da perturbação
        theta = np.cos(k * y_idx)

        # Posição local da interface (Ondulada)
        displacement = safe_amp * theta
        x_local_interface = x_interface_mean + displacement

        # Potencial Tangencial (Linear e contínuo)
        psi_tan = -H0t * y_idx

        for x_idx in range(nx):
            # Coordenada relativa à interface ondulada
            # x_rel < 0: Meio 1 (Invasor/Esquerda)
            # x_rel > 0: Meio 2 (Residente/Direita)
            x_rel = x_idx - x_local_interface

            # --- CÁLCULO ESTÁVEL DO POTENCIAL BASE (H_normal) ---
            # Em vez de um "bico" em x=0, usamos a integral da tanh para suavizar.
            # H(x) ~ tanh(x). Psi(x) ~ ln(cosh(x)).
            # Isso conecta a inclinação H0n1 à esquerda com H0n2 à direita suavemente.
            ln_cosh = np.log(np.cosh(x_rel * inv_w)) * w_smooth
            psi_base_normal = -(H_avg * x_rel + H_diff * ln_cosh)

            # --- CÁLCULO ESTÁVEL DA PERTURBAÇÃO ---
            # Mistura suave (Blending) das exponenciais.
            # Função de peso sigmoidal (0 na esq, 1 na dir)
            weight_R = 0.5 * (1.0 + np.tanh(x_rel * inv_w))
            weight_L = 1.0 - weight_R

            # Exponenciais com trava de segurança (clipping)
            # Evita overflow numérico longe da interface
            arg_L = min(20.0, k * x_rel)  # Exp positiva (cresce com x, válida p/ x<0)
            arg_R = min(20.0, -k * x_rel)  # Exp negativa (cresce com -x, válida p/ x>0)

            # Perturbação Esquerda (Meio 1)
            pert_L = C1 * np.exp(arg_L) * theta

            # Perturbação Direita (Meio 2)
            pert_R = C2 * np.exp(arg_R) * theta

            # Combinação ponderada
            psi_pert_total = (weight_L * pert_L) + (weight_R * pert_R)

            # --- SOMA FINAL ---
            psi[y_idx, x_idx] = psi_base_normal + psi_tan + psi_pert_total

    return psi