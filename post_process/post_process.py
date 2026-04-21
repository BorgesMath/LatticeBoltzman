# post_process.py
import os
import numpy as np
import matplotlib.pyplot as plt
from pyevtk.hl import gridToVTK


# =============================================================================
# 1. SETUP DE DIRETÓRIOS
# =============================================================================
def setup_output_dir(mode_m):
    base_dir = f"st_analise_modo_{mode_m}"
    # Remove as pastas de imagens 2D antigas e mantém apenas VTK e séries temporais
    subdirs = ['vtk', 'series_temporais']

    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        if not os.path.exists(path):
            os.makedirs(path)

    return base_dir


# =============================================================================
# 2. EXPORTAÇÃO ESTRUTURADA (VTK - OTIMIZADA)
# =============================================================================
def export_fields_vtk(phi, psi, rho, u_x, u_y, mode_m, t, base_dir):
    ny, nx = phi.shape

    # Definição das coordenadas da grade cartesiana nodal
    x = np.arange(0, nx + 1, dtype=np.float64)
    y = np.arange(0, ny + 1, dtype=np.float64)
    z = np.array([0.0, 1.0], dtype=np.float64)

    # Transposição para ordem F-contiguous e reshape para 3D (X, Y, Z) exigido pelo VTK
    phi_3d = phi.T.reshape((nx, ny, 1))
    rho_3d = rho.T.reshape((nx, ny, 1))
    psi_3d = psi.T.reshape((nx, ny, 1))

    ux_3d = u_x.T.reshape((nx, ny, 1))
    uy_3d = u_y.T.reshape((nx, ny, 1))
    uz_3d = np.zeros_like(ux_3d)

    # Cálculo do Gradiente Magnético in-situ para visualização
    hy, hx = np.gradient(-psi)
    hx_3d = hx.T.reshape((nx, ny, 1))
    hy_3d = hy.T.reshape((nx, ny, 1))
    hz_3d = np.zeros_like(hx_3d)

    caminho_arquivo = os.path.join(base_dir, 'vtk', f"dados_macro_{t:05d}")

    gridToVTK(
        caminho_arquivo, x, y, z,
        cellData={
            "fase_phi": phi_3d,
            "densidade_rho": rho_3d,
            "potencial_psi": psi_3d,
            "velocidade": (ux_3d, uy_3d, uz_3d),
            "campo_magnetico_H": (hx_3d, hy_3d, hz_3d)
        }
    )


# =============================================================================
# 3. DIAGNÓSTICOS TOPOLÓGICOS E TEMPORAIS (MANTIDOS)
# =============================================================================
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