import sys
import os
import glob
import json
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt


def compute_curvature_field(phi):
    """Calcula o campo de curvatura 2D a partir da fase phi."""
    dy, dx = np.gradient(phi)
    d2y, dy_dx = np.gradient(dy)
    dx_dy, d2x = np.gradient(dx)
    num = (dx ** 2) * d2y + (dy ** 2) * d2x - 2.0 * dx * dy * dx_dy
    den = (dx ** 2 + dy ** 2 + 1e-12) ** 1.5
    return num / den


def main():
    if len(sys.argv) < 2:
        print("Uso: python valida_case.py <diretorio_do_caso>")
        sys.exit(1)

    case_dir = sys.argv[1]
    report_path = os.path.join(case_dir, "relatorio_execucao.json")

    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Relatório não encontrado: {report_path}")

    with open(report_path, 'r', encoding='utf-8') as f:
        log_data = json.load(f)

    params = log_data.get("parametros_contorno", {})
    k_imp = params.get("K_0")
    tau_in = params.get("TAU_IN")
    tau_out = params.get("TAU_OUT")
    sigma_imp = params.get("SIGMA")

    nu_in = (tau_in - 0.5) / 3.0
    nu_out = (tau_out - 0.5) / 3.0

    vtk_files = glob.glob(os.path.join(case_dir, "vtk", "*.vtr"))
    if not vtk_files:
        raise FileNotFoundError("Nenhum arquivo VTK encontrado no diretório.")

    # Avaliação no estado de maior desenvolvimento da instabilidade
    latest_vtk = max(vtk_files, key=os.path.getmtime)
    print(f"A processar o snapshot: {os.path.basename(latest_vtk)}")

    mesh = pv.read(latest_vtk)
    NX_nodes, NY_nodes, _ = mesh.dimensions
    NX, NY = NX_nodes - 1, NY_nodes - 1

    # Extração e reformatação (Fortran-like para C-like)
    phi_2d = mesh.cell_data['fase_phi'].reshape((NX, NY), order='F').T
    rho_2d = mesh.cell_data['densidade_rho'].reshape((NX, NY), order='F').T
    ux_2d = mesh.cell_data['velocidade'][:, 0].reshape((NX, NY), order='F').T

    p_2d = rho_2d / 3.0
    kappa_2d = compute_curvature_field(phi_2d)

    # 1. Localização da Interface
    x_int = np.zeros(NY)
    for y in range(NY):
        crossings = np.where(np.diff(np.sign(phi_2d[y, :])))[0]
        if len(crossings) > 0:
            # Ponto de cruzamento (aproximação linear local)
            x0 = crossings[-1]
            phi0, phi1 = phi_2d[y, x0], phi_2d[y, x0 + 1]
            x_int[y] = x0 - phi0 / (phi1 - phi0 + 1e-15)
        else:
            x_int[y] = NX / 2.0

    min_x_int, max_x_int = int(np.min(x_int)), int(np.max(x_int))
    margin = int(0.10 * NX)  # Margem de segurança para evitar interface difusa

    # 2. Definição das Zonas de Bulk
    idx_bulk_1 = np.arange(margin, min_x_int - margin)
    idx_bulk_2 = np.arange(max_x_int + margin, NX - margin)

    if len(idx_bulk_1) < 5 or len(idx_bulk_2) < 5:
        raise ValueError("Espaço insuficiente no bulk para extrapolação linear.")

    # 3. Análise Linha a Linha (Darcy Local e Salto Capilar)
    dp_cap_list = []
    kappa_list = []
    k_rec_in_list = []
    k_rec_out_list = []

    for y in range(NY):
        # Regressão linear no Bulk 1 (Fluido Injetado)
        p1_poly = np.polyfit(idx_bulk_1, p_2d[y, idx_bulk_1], 1)
        dp1_dx = p1_poly[0]
        u1_mean = np.mean(ux_2d[y, idx_bulk_1])
        rho1_mean = np.mean(rho_2d[y, idx_bulk_1])
        k1 = (nu_in * rho1_mean * u1_mean) / (-dp1_dx + 1e-15)
        k_rec_in_list.append(k1)

        # Regressão linear no Bulk 2 (Fluido Deslocado)
        p2_poly = np.polyfit(idx_bulk_2, p_2d[y, idx_bulk_2], 1)
        dp2_dx = p2_poly[0]
        u2_mean = np.mean(ux_2d[y, idx_bulk_2])
        rho2_mean = np.mean(rho_2d[y, idx_bulk_2])
        k2 = (nu_out * rho2_mean * u2_mean) / (-dp2_dx + 1e-15)
        k_rec_out_list.append(k2)

        # Extrapolação da pressão para a interface (Lei de Young-Laplace Macroscópica)
        p1_extrap = p1_poly[0] * x_int[y] + p1_poly[1]
        p2_extrap = p2_poly[0] * x_int[y] + p2_poly[1]

        dp_cap_list.append(p1_extrap - p2_extrap)

        # Curvatura interpolada
        x_idx = int(np.round(x_int[y]))
        x_idx = np.clip(x_idx, 0, NX - 1)
        kappa_list.append(kappa_2d[y, x_idx])

    # 4. Avaliação Estatística
    k_rec_in_avg = np.mean(k_rec_in_list)
    k_rec_out_avg = np.mean(k_rec_out_list)
    erro_k1 = abs(k_imp - k_rec_in_avg) / k_imp * 100
    erro_k2 = abs(k_imp - k_rec_out_avg) / k_imp * 100

    kappa_arr = np.array(kappa_list)
    dp_cap_arr = np.array(dp_cap_list)

    # Regressão para recuperar Sigma (y = mx + b)
    # Em deslocamentos viscosos, a extrapolação já lida com os gradientes dinâmicos.
    # O declive deve equivaler à tensão superficial numérica.
    coeffs_sigma = np.polyfit(kappa_arr, dp_cap_arr, 1)
    sigma_rec = coeffs_sigma[0]
    erro_sigma = abs(sigma_imp - sigma_rec) / sigma_imp * 100

    # 5. Relatório Técnico
    print(f"\n{'-' * 50}")
    print("VALIDAÇÃO CONJUNTA: LSA / DARCY / YOUNG-LAPLACE")
    print(f"{'-' * 50}")
    print(f"[Darcy] Permeabilidade Imposta : {k_imp:.4f}")
    print(f"[Darcy] K_rec Médio (Bulk In)  : {k_rec_in_avg:.4f} (Erro: {erro_k1:.2f}%)")
    print(f"[Darcy] K_rec Médio (Bulk Out) : {k_rec_out_avg:.4f} (Erro: {erro_k2:.2f}%)")
    print(f"[Laplace] Tensão Imposta       : {sigma_imp:.6f}")
    print(f"[Laplace] Tensão Recuperada    : {sigma_rec:.6f} (Erro: {erro_sigma:.2f}%)")
    print(f"{'-' * 50}\n")

    # 6. Representação Gráfica
    plt.style.use('default')
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 11})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    # Gráfico A: Validação Young-Laplace Local
    ax1.plot(kappa_arr, dp_cap_arr, 'o', color='#1a5276', markersize=4, alpha=0.6, label='Dispersão na Interface')

    k_min, k_max = np.min(kappa_arr), np.max(kappa_arr)
    k_line = np.linspace(k_min, k_max, 100)

    ax1.plot(k_line, sigma_imp * k_line + coeffs_sigma[1], 'k--', lw=1.8,
             label=fr'Teoria ($\sigma_{{imp}}={sigma_imp:.4f}$)')
    ax1.plot(k_line, sigma_rec * k_line + coeffs_sigma[1], 'r-', lw=1.5,
             label=fr'Numérico ($\sigma_{{num}}={sigma_rec:.4f}$)')

    ax1.set_title("Validação Young-Laplace Macroscópica")
    ax1.set_xlabel(r"Curvatura Local $\kappa$ [lu$^{-1}$]")
    ax1.set_ylabel(r"Salto Capilar Extrapolado $\Delta P_{cap}$ [lu]")
    ax1.legend(frameon=True, edgecolor='black')
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Gráfico B: Extração de Darcy (Exemplo numa linha de y=NY/2)
    y_mid = NY // 2
    ax2.plot(np.arange(NX), p_2d[y_mid, :], '-', color='#555', alpha=0.5, label='Perfil de Pressão LBM')

    p1_poly_mid = np.polyfit(idx_bulk_1, p_2d[y_mid, idx_bulk_1], 1)
    p2_poly_mid = np.polyfit(idx_bulk_2, p_2d[y_mid, idx_bulk_2], 1)

    x_extrap_1 = np.arange(margin, int(x_int[y_mid]) + 1)
    x_extrap_2 = np.arange(int(x_int[y_mid]), NX - margin)

    ax2.plot(x_extrap_1, p1_poly_mid[0] * x_extrap_1 + p1_poly_mid[1], 'b--', lw=1.5, label='Extrapolação Bulk In')
    ax2.plot(x_extrap_2, p2_poly_mid[0] * x_extrap_2 + p2_poly_mid[1], 'r--', lw=1.5, label='Extrapolação Bulk Out')
    ax2.axvline(x_int[y_mid], color='black', ls=':', label='Interface')

    ax2.set_title(fr"Perfil de Pressão Extrapolado ($y={y_mid}$)")
    ax2.set_xlabel(r"Posição Longitudinal $x$ [lu]")
    ax2.set_ylabel(r"Pressão $P$ [lu]")
    ax2.legend(frameon=True, edgecolor='black')
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    out_img = os.path.join(case_dir, "validacao_conjunta_lsa.png")
    fig.savefig(out_img, dpi=300)
    print(f"Gráfico guardado em: {out_img}")


if __name__ == "__main__":
    main()