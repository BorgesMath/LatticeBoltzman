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
    psi : potencial de PERTURBAÇÃO (psi_tilde) tal que H_total = H_fundo - grad(psi_tilde)
    Hx_fundo, Hy_fundo : componentes constantes do campo de fundo (H0*cos, H0*sin).
    """
    ny, nx = phi.shape
    x = np.arange(0, nx + 1, dtype=np.float64)
    y = np.arange(0, ny + 1, dtype=np.float64)
    z = np.array([0.0, 1.0], dtype=np.float64)

    phi_3d = phi.T.reshape((nx, ny, 1))
    rho_3d = rho.T.reshape((nx, ny, 1))

    ux_3d = u_x.T.reshape((nx, ny, 1))
    uy_3d = u_y.T.reshape((nx, ny, 1))
    uz_3d = np.zeros_like(ux_3d)

    # Gradiente da perturbação (axis 0 = y, axis 1 = x para array shape (ny, nx))
    dpsi_dy, dpsi_dx = np.gradient(psi)

    # Campo magnético TOTAL: H = H_fundo - grad(psi_tilde)
    hx_total = Hx_fundo - dpsi_dx
    hy_total = Hy_fundo - dpsi_dy

    # Potencial TOTAL para visualização (psi_total = psi_tilde - H_fundo·r)
    xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
    psi_total = psi - (Hx_fundo * xv + Hy_fundo * yv)

    psi_tilde_3d = psi.T.reshape((nx, ny, 1))
    psi_total_3d = psi_total.T.reshape((nx, ny, 1))
    hx_3d = hx_total.T.reshape((nx, ny, 1))
    hy_3d = hy_total.T.reshape((nx, ny, 1))
    hz_3d = np.zeros_like(hx_3d)

    caminho_arquivo = os.path.join(base_dir, 'vtk', f"dados_macro_{t:05d}")

    gridToVTK(
        caminho_arquivo, x, y, z,
        cellData={
            "fase_phi": phi_3d,
            "densidade_rho": rho_3d,
            "potencial_perturbacao_psi_tilde": psi_tilde_3d,
            "potencial_total_psi": psi_total_3d,
            "velocidade": (ux_3d, uy_3d, uz_3d),
            "campo_magnetico_H": (hx_3d, hy_3d, hz_3d)
        }
    )


# =============================================================================
# 3. DIAGNÓSTICOS TEMPORAIS
# =============================================================================
def export_time_series(mass_history, time_steps, base_dir):
    plt.figure(figsize=(8, 4))
    plt.plot(time_steps, mass_history, color='blue', linewidth=1.5)
    plt.title("Conservação da Massa Total")
    plt.ylabel(r"$\sum \rho$")
    plt.xlabel("Iterações (t)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'series_temporais', "massa_total.png"), dpi=150)
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

def export_simulation_log(params, mass_history, exec_duration, base_dir):
    """
    Gera e exporta o relatório JSON de integridade e diagnósticos da simulação LBM.
    Incorpora relaxação do limite de conservação de massa (5%) para regimes compressíveis de Darcy
    e avaliação do Número de Reynolds no Poro (Re_K) para detecção do regime de Forchheimer.
    Análise morfológica (curvatura, amplitude, regime) delegada a resultado_curvatura_temporal.py.
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

    # Grupos adimensionais
    nu_out = (params["TAU_OUT"] - 0.5) / 3.0
    sigma = params["SIGMA"]
    ny = params["NY"]
    nx = params["NX"]

    ca = (nu_kinematic * params["U_INLET"]) / sigma if sigma > 0 else float("inf")
    da = params["K_0"] / (ny ** 2)
    re = (params["U_INLET"] * ny) / nu_kinematic
    m_visc = nu_out / nu_kinematic
    cn = params["INTERFACE_WIDTH"] / ny
    ar = nx / ny
    h0 = params.get("H0", 0.0)
    chi_max = params.get("CHI_MAX", 0.0)
    bo_mag = (chi_max * h0 ** 2 * ny) / sigma if (sigma > 0 and h0 > 0) else 0.0

    log_data = {
        "identificacao": {
            "id_caso": params["id_caso"],
            "status_execucao": status,
            "tempo_execucao_segundos": round(exec_duration, 2),
            "tempo_execucao_minutos": round(exec_duration / 60.0, 2)
        },
        "diagnostico_fisico": {
            "reynolds_poro_rek": float(round(re_k, 4)),
            "nota_morfologica": "Curvatura e amplitude analisadas por resultado_curvatura_temporal.py"
        },
        "integridade_numerica": {
            "massa_total_t0": float(mass_t0) if not np.isnan(mass_t0) else "NaN",
            "massa_total_t_final": float(mass_tf) if not np.isnan(mass_tf) else "NaN",
            "erro_conservacao_massa_percentual": float(round(mass_var_pct, 6)) if not np.isnan(mass_var_pct) else "NaN",
            "alertas_identificados": alertas
        },
        "parametros_adimensionais": {
            "Ca":     round(ca, 6),
            "Da":     round(da, 6),
            "Re":     round(re, 6),
            "Re_K":   float(round(re_k, 6)),
            "M_visc": round(m_visc, 6),
            "Cn":     round(cn, 6),
            "AR":     round(ar, 4),
            "Bo_mag": round(bo_mag, 6)
        },
        "parametros_contorno": params
    }

    log_path = os.path.join(base_dir, "relatorio_execucao.json")
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=4, ensure_ascii=False)


# =============================================================================
# 5. VALIDAÇÃO DE DARCY (escoamento unifásico, CH_SUBSTEPS=0, mode_m=0)
# =============================================================================
def validate_darcy_flow(rho, params, base_dir):
    """
    Compara o perfil de pressão simulado com a previsão analítica de Darcy:
        dp/dx = -(nu / K) * U_inlet
    Em LBM: p = rho / 3.  Plota perfil de pressão e gradiente, salva PNG e
    imprime o erro relativo médio no terminal.
    """
    nu_in = (params["TAU_IN"] - 0.5) / 3.0
    k0    = params["K_0"]
    u_in  = params["U_INLET"]
    nx    = rho.shape[1]

    # Pressão (LBM): p = cs² × rho = rho / 3
    p_mean = rho.mean(axis=0) / 3.0
    x = np.arange(nx, dtype=np.float64)

    # Gradiente numérico de pressão (central, borda por diferença simples)
    dp_dx_sim = np.gradient(p_mean, x)

    # Previsão analítica de Darcy (pressão cai da entrada para a saída)
    dp_dx_theory = -(nu_in / k0) * u_in

    # Erro relativo médio (ignora 5 células de contorno em cada extremidade)
    interior = dp_dx_sim[5:-5]
    err_pct = (np.mean(np.abs(interior - dp_dx_theory))
               / (abs(dp_dx_theory) + 1e-12) * 100.0)

    # Perfil teórico linear ancorado no outlet (x = nx-1, p = 1/3)
    p_theory = 1.0 / 3.0 + dp_dx_theory * (x - (nx - 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(x, p_mean,   color='#1a5276', linewidth=1.5, label="Simulado")
    ax1.plot(x, p_theory, color='#922b21', linewidth=1.2, linestyle='--',
             label="Darcy analítico")
    ax1.set_xlabel(r"$x$ (l.u.)")
    ax1.set_ylabel(r"$p = \rho/3$ (l.u.)")
    ax1.set_title("Perfil de Pressão")
    ax1.legend(frameon=True, fancybox=False, edgecolor='black')
    ax1.tick_params(direction='in', top=True, right=True)

    ax2.plot(x[5:-5], dp_dx_sim[5:-5], color='#1a5276', linewidth=1.5,
             label=r"$dp/dx$ simulado")
    ax2.axhline(dp_dx_theory, color='#922b21', linestyle='--', linewidth=1.2,
                label=r"$dp/dx$ teórico")
    ax2.set_xlabel(r"$x$ (l.u.)")
    ax2.set_ylabel(r"$dp/dx$ (l.u.)")
    ax2.set_title(f"Gradiente de Pressão  (erro médio = {err_pct:.2f} %)")
    ax2.legend(frameon=True, fancybox=False, edgecolor='black')
    ax2.tick_params(direction='in', top=True, right=True)

    fig.tight_layout()
    out_path = os.path.join(base_dir, "validacao_darcy.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"[Darcy] Erro relativo médio dp/dx: {err_pct:.3f}%")
    print(f"[Darcy] Gráfico salvo em: {out_path}")