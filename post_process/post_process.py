# post_process.py
import os
import numpy as np
import matplotlib.pyplot as plt


def setup_output_dir(mode_m):
    base_dir = f"st_analise_modo_{mode_m}"
    # ADICIONADO DIRETÓRIO DE PRESSÃO
    subdirs = ['fase', 'velocidade', 'densidade', 'magnetico', 'pressao', 'series_temporais']

    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        if not os.path.exists(path):
            os.makedirs(path)

    return base_dir


def compute_interface_curvature(phi):
    dy, dx = np.gradient(phi)
    d2y, dy_dx = np.gradient(dy)
    dx_dy, d2x = np.gradient(dx)

    num = (dx ** 2 * d2y) + (dy ** 2 * d2x) - (2.0 * dx * dy * dx_dy)
    den = (dx ** 2 + dy ** 2 + 1e-8) ** 1.5
    kappa = num / den

    interface_mask = np.abs(phi) < 0.1
    if np.any(interface_mask):
        return np.mean(np.abs(kappa[interface_mask]))
    return 0.0


def export_fields(phi, psi, rho, u_x, u_y, mode_m, t, base_dir):
    # 1. Campo de Fase
    plt.figure(figsize=(8, 4))
    im0 = plt.imshow(phi, cmap='RdBu', origin='lower')
    plt.contour(phi, levels=[0], colors='black', linewidths=1)
    plt.title(fr"Campo de Fase ($\phi$) - t={t}")
    plt.colorbar(im0, fraction=0.046, pad=0.04)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'fase', f"fase_{t:05d}.png"), dpi=150)
    plt.close()

    # 2. Campo Vetorial de Velocidade
    plt.figure(figsize=(8, 4))
    u_mag = np.sqrt(u_x ** 2 + u_y ** 2)
    im1 = plt.imshow(u_mag, cmap='viridis', origin='lower')
    Y, X = np.mgrid[0:u_x.shape[0], 0:u_x.shape[1]]
    plt.streamplot(X, Y, u_x, u_y, color='white', density=1.0, linewidth=0.5)
    plt.title(fr"Velocidade Macroscópica ($|\mathbf{{u}}|$ e streamlines) - t={t}")
    plt.colorbar(im1, fraction=0.046, pad=0.04)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'velocidade', f"vel_{t:05d}.png"), dpi=150)
    plt.close()

    # 3. Campo de Densidade
    plt.figure(figsize=(8, 4))
    im2 = plt.imshow(rho, cmap='plasma', origin='lower')
    plt.title(fr"Densidade do Fluido ($\rho$) - t={t}")
    plt.colorbar(im2, fraction=0.046, pad=0.04)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'densidade', f"rho_{t:05d}.png"), dpi=150)
    plt.close()

    # 4. Campo Magnético
    plt.figure(figsize=(8, 4))
    hy, hx = np.gradient(-psi)
    h_mag = np.sqrt(hx ** 2 + hy ** 2)
    im3 = plt.imshow(h_mag, cmap='magma', origin='lower')
    plt.streamplot(X, Y, hx, hy, color='cyan', density=0.8, linewidth=0.5)
    plt.title(fr"Campo Magnético ($|\mathbf{{H}}|$ e streamlines) - t={t}")
    plt.colorbar(im3, fraction=0.046, pad=0.04)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'magnetico', f"mag_{t:05d}.png"), dpi=150)
    plt.close()

    # 5. NOVO: Campo de Pressão (Equação de Estado LBM)
    plt.figure(figsize=(8, 4))
    pressao = rho / 3.0  # P = rho * cs^2
    im4 = plt.imshow(pressao, cmap='inferno', origin='lower')
    plt.title(fr"Pressão Macroscópica ($P = \rho / 3$) - t={t}")
    plt.colorbar(im4, fraction=0.046, pad=0.04)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'pressao', f"pressao_{t:05d}.png"), dpi=150)
    plt.close()


def export_time_series(mass_history, curv_history, time_steps, mode_m, base_dir):
    plt.figure(figsize=(8, 4))
    plt.plot(time_steps, mass_history, color='blue', linewidth=1.5)
    plt.title("Conservação da Massa Total")
    plt.ylabel(r"$\sum \rho$")
    plt.xlabel("Iterações (t)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'series_temporais', f"massa_total_modo_{mode_m}.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(time_steps, curv_history, color='red', linewidth=1.5)
    plt.title("Curvatura Média Absoluta da Interface")
    plt.ylabel(r"$\bar{|\kappa|}$ interfacial")
    plt.xlabel("Iterações (t)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'series_temporais', f"curvatura_modo_{mode_m}.png"), dpi=150)
    plt.close()


def export_tip_position(phi, mode_m, base_dir):
    ny, nx = phi.shape
    indices = np.where(phi > 0.0)
    if indices[1].size > 0:
        max_x = np.max(indices[1])
    else:
        max_x = 0

    path = os.path.join(base_dir, "tip_position.txt")
    with open(path, 'w') as f:
        f.write(str(max_x))


def export_growth_rate(amp_history, time_steps, mode_m, s_teorico, base_dir):
    plt.figure(figsize=(8, 6))

    # CORREÇÃO: Descarta estritamente os primeiros 10% da simulação (inércia/transiente)
    idx_10_percent = int(len(time_steps) * 0.10)
    t_corte = time_steps[idx_10_percent:]
    amp_corte = amp_history[idx_10_percent:]

    # Filtra ruídos numéricos
    valid_mask = amp_corte > 1e-4
    t_valid = t_corte[valid_mask]
    amp_valid = amp_corte[valid_mask]

    if len(amp_valid) == 0:
        print("Aviso: Amplitude insuficiente para gráfico de crescimento.")
        return

    amp_0 = amp_valid[0]
    ln_A_A0 = np.log(amp_valid / amp_0)

    # Plot Numérico (apenas região pós-10%)
    plt.plot(t_valid, ln_A_A0, 'b-', linewidth=2, label=r'LBM Numérico')

    # Plot Teórico
    t_shifted = t_valid - t_valid[0]
    reta_teorica = s_teorico * t_shifted
    plt.plot(t_valid, reta_teorica, 'r--', linewidth=2, label=rf'LSA Teórica ($s = {s_teorico:.2e}$)')

    # Extração Numérica Final (Regressão sobre o terço inicial da região válida para capturar o limite linear puro)
    limit_idx = max(2, int(len(t_valid) * 0.33))
    t_linear = t_valid[:limit_idx]
    ln_linear = ln_A_A0[:limit_idx]

    if len(t_linear) > 1:
        s_num_final, intercept = np.polyfit(t_linear - t_linear[0], ln_linear, 1)
        plt.plot(t_linear, (s_num_final * (t_linear - t_linear[0]) + intercept), 'g:', linewidth=3,
                 label=rf'Ajuste Final LBM ($s_{{num}} = {s_num_final:.2e}$)')

        erro_final = abs(s_num_final - s_teorico) / abs(s_teorico) * 100 if s_teorico != 0 else 0
        plt.title(f"Validação: Taxa de Crescimento Consolidada (Modo {mode_m})\n"
                  f"Erro Relativo (s_num vs s_teo): {erro_final:.2f}% (Primeiros 10% ignorados)")
    else:
        plt.title(f"Validação: Taxa de Crescimento (Modo {mode_m})")

    plt.xlabel("Iterações (t)")
    plt.ylabel(r"$\ln(\lambda(t) / \lambda_0)$")
    plt.legend()
    plt.grid(True, alpha=0.5, linestyle=':')

    path = os.path.join(base_dir, 'series_temporais', f"validacao_crescimento_modo_{mode_m}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    if len(t_linear) > 1:
        print(f"\n[Validação Física Final] Modo: {mode_m}")
        print(f"Taxa Numérica Extraída: {s_num_final:.6e}")
        print(f"Taxa Teórica Esperada:  {s_teorico:.6e}")
        print(f"Erro Relativo OLS:      {erro_final:.2f}%")