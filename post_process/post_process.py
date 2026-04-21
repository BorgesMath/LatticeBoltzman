# post_process.py
import os
import numpy as np
import matplotlib.pyplot as plt
from pyevtk.hl import gridToVTK
from datetime import datetime
import json


# =============================================================================
# 1. SETUP DE DIRETÓRIOS
# =============================================================================
def setup_output_dir(id_caso):
    agora = datetime.now()
    timestamp = agora.strftime("d%dmes%m-h%H_min%M")
    base_dir = f"{id_caso}_{timestamp}"

    subdirs = ['vtk', 'series_temporais']

    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        if not os.path.exists(path):
            os.makedirs(path)

    return base_dir


# =============================================================================
# 2. EXPORTAÇÃO ESTRUTURADA (VTK - OTIMIZADA)
# =============================================================================
def export_fields_vtk(phi, psi, rho, u_x, u_y, t, base_dir):
    ny, nx = phi.shape
    x = np.arange(0, nx + 1, dtype=np.float64)
    y = np.arange(0, ny + 1, dtype=np.float64)
    z = np.array([0.0, 1.0], dtype=np.float64)

    phi_3d = phi.T.reshape((nx, ny, 1))
    rho_3d = rho.T.reshape((nx, ny, 1))
    psi_3d = psi.T.reshape((nx, ny, 1))

    ux_3d = u_x.T.reshape((nx, ny, 1))
    uy_3d = u_y.T.reshape((nx, ny, 1))
    uz_3d = np.zeros_like(ux_3d)

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


def export_time_series(mass_history, curv_history, time_steps, base_dir):
    plt.figure(figsize=(8, 4))
    plt.plot(time_steps, mass_history, color='blue', linewidth=1.5)
    plt.title("Conservação da Massa Total")
    plt.ylabel(r"$\sum \rho$")
    plt.xlabel("Iterações (t)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'series_temporais', "massa_total.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(time_steps, curv_history, color='red', linewidth=1.5)
    plt.title("Curvatura Média Absoluta da Interface")
    plt.ylabel(r"$\bar{|\kappa|}$ interfacial")
    plt.xlabel("Iterações (t)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'series_temporais', "curvatura.png"), dpi=150)
    plt.close()


def export_tip_position(phi, base_dir):
    indices = np.where(phi > 0.0)
    if indices[1].size > 0:
        max_x = np.max(indices[1])
    else:
        max_x = 0

    path = os.path.join(base_dir, "tip_position.txt")
    with open(path, 'w') as f:
        f.write(str(max_x))


# =============================================================================
# 4. CONSTRUCAO LOG
# =============================================================================

def export_simulation_log(params, mass_history, curv_history, exec_time, base_dir):
    """
    Gera um relatório diagnóstico da estabilidade termodinâmica e integridade numérica.
    """
    mass_initial = mass_history[0]
    mass_final = mass_history[-1]

    # O LBM com Cahn-Hilliard requer conservação estrita da massa global
    if mass_initial == 0:
        mass_variation_pct = 0.0
    else:
        mass_variation_pct = ((mass_final - mass_initial) / mass_initial) * 100.0

    # Diagnóstico empírico de estabilidade morfológica
    # Se a curvatura cresce consistentemente, a interface é instável (Saffman-Taylor fingering)
    curv_growth = curv_history[-1] / (curv_history[0] + 1e-12)
    if curv_growth > 1.1:
        regime_morfologico = "Instável (Crescimento de perturbação / Fingering detectado)"
    else:
        regime_morfologico = "Estável (Supressão magnética/viscosa ou interface plana)"

    status = "BEM SUCEDIDA"
    alertas = []

    # Verificações de falha e instabilidade numérica
    if np.isnan(mass_final) or np.isnan(curv_history[-1]):
        status = "FALHA CRÍTICA"
        alertas.append(
            "Divergência Numérica (NaN) detectada nos tensores. Condição de Courant (CFL) ou limite de estabilidade do operador de colisão violados.")

    if abs(mass_variation_pct) > 0.5:
        if status != "FALHA CRÍTICA": status = "ALERTA"
        alertas.append(
            f"Violação da conservação de massa ({mass_variation_pct:.4f}%). Recomenda-se reduzir dt (aumentar CH_SUBSTEPS) ou ajustar M_MOBILITY.")

    if curv_history[-1] > 20.0:
        if status != "FALHA CRÍTICA": status = "ALERTA"
        alertas.append(
            "Curvatura interfacial extrema. Possível artefato de grade (grid pinning) ou ramificação severa (tip-splitting). Verificar refinamento da malha espacial (NX, NY).")

    log_data = {
        "identificacao": {
            "id_caso": params.get("id_caso", "N/A"),
            "status_execucao": status,
            "tempo_execucao_segundos": round(exec_time, 2),
            "tempo_execucao_minutos": round(exec_time / 60.0, 2)
        },
        "diagnostico_fisico": {
            "regime_morfologico": regime_morfologico,
            "razao_crescimento_curvatura": round(curv_growth, 6),
            "curvatura_t0": curv_history[0],
            "curvatura_t_final": curv_history[-1]
        },
        "integridade_numerica": {
            "massa_total_t0": mass_initial,
            "massa_total_t_final": mass_final,
            "erro_conservacao_massa_percentual": round(mass_variation_pct, 6),
            "alertas_identificados": alertas
        },
        "parametros_contorno": params
    }

    caminho_arquivo = os.path.join(base_dir, "relatorio_execucao.json")
    with open(caminho_arquivo, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=4, ensure_ascii=False)