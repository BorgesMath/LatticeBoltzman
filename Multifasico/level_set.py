import numpy as np


def step_advection_upwind(phi, u_x, u_y, dx=1.0, dt=1.0):
    """
    Integração temporal da equação de advecção via Upwind de 1ª ordem.
    Garante estabilidade condicional de Courant-Friedrichs-Lewy (CFL).
    """
    # Diferenças espaciais Regressivas (Backward) e Progressivas (Forward)
    phi_bx = (phi - np.roll(phi, 1, axis=1)) / dx
    phi_fx = (np.roll(phi, -1, axis=1) - phi) / dx
    phi_by = (phi - np.roll(phi, 1, axis=0)) / dx
    phi_fy = (np.roll(phi, -1, axis=0) - phi) / dx

    # Seleção de fluxo baseada na direção da característica (u_x, u_y)
    dphi_dx = np.where(u_x > 0, phi_bx, phi_fx)
    dphi_dy = np.where(u_y > 0, phi_by, phi_fy)

    # Integração Euler Explícito
    return phi - dt * (u_x * dphi_dx + u_y * dphi_dy)


def reinitialize_godunov(phi, dx=1.0, dtau=0.5, steps=5, epsilon_sign=1.0):
    """
    Reinicialização da SDF resolvendo a Equação de Eikonal relaxada.
    Utiliza o esquema de Godunov para estabilidade entrópica do Hamiltoniano.
    """
    phi_new = np.copy(phi)
    # Função sinal suavizada (evita singularidades em phi=0)
    S = phi / np.sqrt(phi ** 2 + epsilon_sign ** 2)

    for _ in range(steps):
        a = (phi_new - np.roll(phi_new, 1, axis=1)) / dx  # D_x^-
        b = (np.roll(phi_new, -1, axis=1) - phi_new) / dx  # D_x^+
        c = (phi_new - np.roll(phi_new, 1, axis=0)) / dx  # D_y^-
        d = (np.roll(phi_new, -1, axis=0) - phi_new) / dx  # D_y^+

        # Fluxo para características divergentes do plano zero (S > 0)
        grad_plus = np.sqrt(
            np.maximum(np.maximum(a, 0) ** 2, np.minimum(b, 0) ** 2) +
            np.maximum(np.maximum(c, 0) ** 2, np.minimum(d, 0) ** 2)
        )

        # Fluxo para características convergentes ao plano zero (S < 0)
        grad_minus = np.sqrt(
            np.maximum(np.minimum(a, 0) ** 2, np.maximum(b, 0) ** 2) +
            np.maximum(np.minimum(c, 0) ** 2, np.maximum(d, 0) ** 2)
        )

        # Seleção do gradiente de Godunov
        grad_phi = np.where(S > 0, grad_plus, grad_minus)

        # Atualização temporal fictícia
        phi_new = phi_new - dtau * S * (grad_phi - 1.0)

    return phi_new


def compute_curvature_and_dirac(phi, epsilon=1.5, dx=1.0):
    """
    Computa a curvatura topológica K e a Delta de Dirac regularizada.
    Necessário para o cálculo da força de Tensão Superficial e mapeamento de viscosidade.
    """
    # Gradientes centrais isotrópicos
    phi_x = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2.0 * dx)
    phi_y = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2.0 * dx)

    phi_xx = (np.roll(phi, -1, axis=1) - 2.0 * phi + np.roll(phi, 1, axis=1)) / (dx ** 2)
    phi_yy = (np.roll(phi, -1, axis=0) - 2.0 * phi + np.roll(phi, 1, axis=0)) / (dx ** 2)
    phi_xy = (np.roll(np.roll(phi, -1, axis=1), -1, axis=0) -
              np.roll(np.roll(phi, 1, axis=1), -1, axis=0) -
              np.roll(np.roll(phi, -1, axis=1), 1, axis=0) +
              np.roll(np.roll(phi, 1, axis=1), 1, axis=0)) / (4.0 * dx ** 2)

    grad_mag_sq = phi_x ** 2 + phi_y ** 2 + 1e-12  # Previne divisão por zero
    grad_mag = np.sqrt(grad_mag_sq)

    # Curvatura Kappa = divergente do vetor normal unitário
    kappa = (phi_xx * phi_y ** 2 - 2.0 * phi_xy * phi_x * phi_y + phi_yy * phi_x ** 2) / (grad_mag_sq * grad_mag)

    # Delta de Dirac regularizada (suporte compacto [-epsilon, epsilon])
    dirac = np.zeros_like(phi)
    mask = np.abs(phi) <= epsilon
    dirac[mask] = (1.0 / (2.0 * epsilon)) * (1.0 + np.cos(np.pi * phi[mask] / epsilon))

    return kappa, dirac, phi_x, phi_y