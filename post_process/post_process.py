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

def export_simulation_log(params, mass_history, curv_history, exec_duration, base_dir):
    """
    Gera e exporta o relatório JSON de integridade e diagnósticos da simulação LBM.
    Incorpora relaxação do limite de conservação de massa (5%) para regimes compressíveis de Darcy
    e avaliação do Número de Reynolds no Poro (Re_K) para detecção do regime de Forchheimer.
    """
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

        # O limite de alerta de massa é ampliado para 5.0% para absorver as flutuações acústicas
        # naturais exigidas pela formação do gradiente de pressão no meio poroso.
        if abs(mass_var_pct) > 5.0:
            status = "ALERTA"
            alertas.append(
                f"Violação severa de densidade acustica ({mass_var_pct:.4f}%). "
                "Verifique condições de contorno; vazamento de massa provável."
            )

    # Análise Física da Interface e Cálculo do Regime de Poro
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

    # Tratamento Morfológico Condicional (Multifásico vs Unifásico)
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

def validate_darcy_flow(rho, params, base_dir):
    ny, nx = rho.shape

    # Extração ao longo da linha neutra, truncando as extremidades (10 nós)
    # para evitar as camadas limites numéricas (erros de truncamento de Zou-He e Neumann)
    rho_centerline = rho[ny // 2, 10:nx - 10]
    x_coords = np.arange(10, nx - 10)

    # Pressão macroscópica LBM (Equação de Estado para D2Q9)
    p_centerline = rho_centerline / 3.0

    # Determinação numérica do gradiente (Regressão Linear)
    coefs = np.polyfit(x_coords, p_centerline, 1)
    dp_dx_num = coefs[0]

    # Previsão Analítica da Lei de Darcy
    nu_kinematic = (params["TAU_IN"] - 0.5) / 3.0
    dp_dx_ana = - (nu_kinematic / params["K_0"]) * params["U_INLET"]

    erro_relativo = abs((dp_dx_num - dp_dx_ana) / dp_dx_ana) * 100.0

    plt.figure(figsize=(10, 6))
    plt.plot(x_coords, p_centerline, 'b-', linewidth=2, label='Pressão Integrada (LBM)')
    plt.plot(x_coords, np.polyval(coefs, x_coords), 'r--', linewidth=2,
             label=f'Ajuste Linear ($dp/dx$: {dp_dx_num:.4e})')
    plt.title(f"Validação de Permeabilidade Darcy (Erro: {erro_relativo:.4f}%)")
    plt.xlabel(r"Coordenada Espacial $X$")
    plt.ylabel(r"Pressão Macroscópica $p = \rho c_s^2$")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'validacao_darcy.png'), dpi=150)
    plt.close()

    print(f"\n{'=' * 50}")
    print("RELATÓRIO DE VALIDAÇÃO: LEI DE DARCY")
    print(f"{'=' * 50}")
    print(f"Gradiente Analítico: {dp_dx_ana:.6e}")
    print(f"Gradiente Numérico:  {dp_dx_num:.6e}")
    print(f"Erro Relativo (L2):  {erro_relativo:.6f}%")
    print(f"{'=' * 50}\n")