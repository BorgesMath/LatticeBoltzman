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
def export_fields_vtk(phi, psi, rho, u_x, u_y, t, base_dir, Hx_fundo=0.0, Hy_fundo=0.0):
    """
    Exporta os dados macroscópicos para VTK.
    Reconstrói o Campo Magnético Total aplicando a superposição do campo de fundo
    com o gradiente do potencial de distúrbio.
    """
    ny, nx = phi.shape
    x = np.arange(0, nx + 1, dtype=np.float64)
    y = np.arange(0, ny + 1, dtype=np.float64)
    z = np.array([0.0, 1.0], dtype=np.float64)

    # Reshape de campos escalares básicos
    phi_3d = phi.T.reshape((nx, ny, 1))
    rho_3d = rho.T.reshape((nx, ny, 1))
    psi_3d = psi.T.reshape((nx, ny, 1))

    # Reshape de vetores de velocidade
    ux_3d = u_x.T.reshape((nx, ny, 1))
    uy_3d = u_y.T.reshape((nx, ny, 1))
    uz_3d = np.zeros_like(ux_3d)

    # Cálculo dos gradientes de perturbação via diferenças finitas centrais
    # np.gradient retorna (derivada_y, derivada_x) para matrizes 2D
    dpsi_dy, dpsi_dx = np.gradient(psi)

    # Vetores do Campo Magnético Induzido (Perturbação)
    hx_induzido = -dpsi_dx
    hy_induzido = -dpsi_dy

    # Vetores do Campo Magnético Total (Fundo + Induzido)
    hx_total = Hx_fundo + hx_induzido
    hy_total = Hy_fundo + hy_induzido

    # Reshape das matrizes magnéticas para o padrão 3D do VTK
    hx_ind_3d = hx_induzido.T.reshape((nx, ny, 1))
    hy_ind_3d = hy_induzido.T.reshape((nx, ny, 1))

    hx_tot_3d = hx_total.T.reshape((nx, ny, 1))
    hy_tot_3d = hy_total.T.reshape((nx, ny, 1))

    hz_3d = np.zeros_like(hx_ind_3d)

    caminho_arquivo = os.path.join(base_dir, 'vtk', f"dados_macro_{t:05d}")

    gridToVTK(
        caminho_arquivo, x, y, z,
        cellData={
            "fase_phi": phi_3d,
            "densidade_rho": rho_3d,
            "potencial_psi_disturbio": psi_3d,
            "velocidade": (ux_3d, uy_3d, uz_3d),
            "campo_magnetico_induzido": (hx_ind_3d, hy_ind_3d, hz_3d),
            "campo_magnetico_TOTAL": (hx_tot_3d, hy_tot_3d, hz_3d)
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

def export_simulation_log(params, mass_history, curv_history, exec_duration, base_dir):
    mass_t0 = mass_history[0]
    mass_tf = mass_history[-1]

    if np.isnan(mass_tf):
        mass_var_pct = np.nan
        status = "FALHA CRÍTICA"
        alertas = [
            "Divergência Numérica (NaN) detectada nos tensores. "
            "Condição de Courant (CFL) ou limite de estabilidade do operador de colisão violados."
        ]
    else:
        mass_var_pct = ((mass_tf - mass_t0) / mass_t0) * 100.0
        status = "BEM SUCEDIDA"
        alertas = []

        if abs(mass_var_pct) > 5.0:
            status = "ALERTA"
            alertas.append(
                f"Violação severa de densidade acustica ({mass_var_pct:.4f}%). "
                "Verifique condições de contorno; vazamento de massa provável."
            )

    nu_kinematic = (params["TAU_IN"] - 0.5) / 3.0
    re_k = (params["U_INLET"] * np.sqrt(params["K_0"])) / nu_kinematic

    if re_k > 1.0:
        if status != "FALHA CRÍTICA":
            status = "ALERTA"
        alertas.append(
            f"Regime de Forchheimer Detectado (Re_K = {re_k:.2f} > 1.0). "
            "A inércia do escoamento não é mais desprezível e o desvio em relação à Lei de Darcy linear excederá 5%. "
            "Para manter a estrita linearidade, reduza U_INLET ou K_0."
        )

    if params.get("CH_SUBSTEPS", 0) > 0:
        curv_t0 = curv_history[0]
        curv_tf = curv_history[-1]
        growth_ratio = curv_tf / curv_t0 if curv_t0 != 0 else 0.0

        if growth_ratio > 1.05:
            regime = "Instável (Crescimento de perturbação / Fingering detectado)"
        else:
            regime = "Estável (Supressão magnética/viscosa ou interface plana)"
    else:
        curv_t0 = 0.0
        curv_tf = 0.0
        growth_ratio = 0.0
        regime = "Escoamento Unifásico (Validação de Meio Poroso)"

    log_data = {
        "identificacao": {
            "id_caso": params["id_caso"],
            "status_execucao": status,
            "tempo_execucao_segundos": round(exec_duration, 2),
            "tempo_execucao_minutos": round(exec_duration / 60.0, 2)
        },
        "diagnostico_fisico": {
            "regime_morfologico": regime,
            "razao_crescimento_curvatura": float(growth_ratio) if not np.isnan(growth_ratio) else "NaN",
            "curvatura_t0": float(curv_t0) if not np.isnan(curv_t0) else "NaN",
            "curvatura_t_final": float(curv_tf) if not np.isnan(curv_tf) else "NaN",
            "reynolds_poro_rek": float(round(re_k, 4))
        },
        "integridade_numerica": {
            "massa_total_t0": float(mass_t0) if not np.isnan(mass_t0) else "NaN",
            "massa_total_t_final": float(mass_tf) if not np.isnan(mass_tf) else "NaN",
            "erro_conservacao_massa_percentual": float(round(mass_var_pct, 6)) if not np.isnan(mass_var_pct) else "NaN",
            "alertas_identificados": alertas
        },
        "parametros_contorno": params
    }

    log_path = os.path.join(base_dir, "relatorio_execucao.json")
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=4, ensure_ascii=False)


def compute_interface_amplitude(phi):
    ny, nx = phi.shape
    interface_x = np.zeros(ny)

    for y in range(ny):
        interface_x[y] = np.argmax(phi[y, :] < 0.0)

    return (np.max(interface_x) - np.min(interface_x)) / 2.0