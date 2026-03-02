# post_process.py
import os
import numpy as np
import matplotlib.pyplot as plt


def setup_output_dir(mode_m):
    base_dir = f"st_analise_modo_{mode_m}"
    subdirs = ['fase', 'velocidade', 'densidade', 'magnetico', 'series_temporais']

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


def export_time_series(mass_history, curv_history, time_steps, mode_m, base_dir):
    # Gráfico da Massa Total
    plt.figure(figsize=(8, 4))
    plt.plot(time_steps, mass_history, color='blue', linewidth=1.5)
    plt.title("Conservação da Massa Total")
    plt.ylabel(r"$\sum \rho$")
    plt.xlabel("Iterações (t)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'series_temporais', f"massa_total_modo_{mode_m}.png"), dpi=150)
    plt.close()

    # Gráfico da Curvatura Interfacial
    plt.figure(figsize=(8, 4))
    plt.plot(time_steps, curv_history, color='red', linewidth=1.5)
    plt.title("Curvatura Média Absoluta da Interface")
    plt.ylabel(r"$\bar{|\kappa|}$ interfacial")
    plt.xlabel("Iterações (t)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'series_temporais', f"curvatura_modo_{mode_m}.png"), dpi=150)
    plt.close()