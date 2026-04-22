import os
import glob
import json
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

BASE_DIR = os.getcwd()


def process_laplace_results():
    report_files = glob.glob(os.path.join(BASE_DIR, "*", "relatorio_execucao.json"))

    inv_r_list = []
    dp_num_list = []
    sigma_imp = None

    print(f"Encontradas {len(report_files)} simulações com relatórios.")

    for report_path in report_files:
        folder = os.path.dirname(report_path)

        with open(report_path, 'r', encoding='utf-8') as f:
            log_data = json.load(f)

        params = log_data.get("parametros_contorno", {})
        mode_m = params.get("mode_m", 0)
        radius_imp = params.get("amplitude")
        current_sigma = params.get("SIGMA")

        if mode_m != -1 or radius_imp is None:
            continue

        sigma_imp = current_sigma

        vtk_files = glob.glob(os.path.join(folder, "vtk", "*.vtr"))
        if not vtk_files:
            continue
        latest_vtk = max(vtk_files, key=os.path.getmtime)

        try:
            mesh = pv.read(latest_vtk)
            NX_nodes, NY_nodes, _ = mesh.dimensions
            NX, NY = NX_nodes - 1, NY_nodes - 1
            rho_flat = mesh.cell_data['densidade_rho']
            rho_2d = rho_flat.reshape((NX, NY), order='F').T
        except Exception as e:
            print(f"Erro em {folder}: {e}")
            continue

        p_2d = rho_2d / 3.0

        # Pressão no interior da gota (Centroide)
        p_in = p_2d[NY // 2, NX // 2]

        # Pressão no bulk do fluido matriz (Canto da grade para evitar interface)
        p_out = p_2d[5, 5]

        dp = p_in - p_out

        inv_r_list.append(1.0 / radius_imp)
        dp_num_list.append(dp)

        print(f"Gota R={radius_imp:4.1f} | Δp = {dp:6.2e} | P_in = {p_in:6.4f} | P_out = {p_out:6.4f}")

    if not inv_r_list:
        print("\nNenhum caso válido da Lei de Young-Laplace (mode_m: -1) foi encontrado.")
        return

    # Regressão Linear: Δp = σ * (1/R) -> O coeficiente angular é a Tensão Superficial recuperada
    inv_r_arr = np.array(inv_r_list)
    dp_arr = np.array(dp_num_list)

    # polyfit forçando intercepto em 0 (y = m*x)
    sigma_rec = np.linalg.lstsq(inv_r_arr[:, np.newaxis], dp_arr, rcond=None)[0][0]

    erro_relativo = abs(sigma_imp - sigma_rec) / sigma_imp * 100.0

    # Configuração de Gráfico Acadêmico
    plt.style.use('default')
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    plt.figure(figsize=(8, 6))

    x_line = np.linspace(0, max(inv_r_arr) * 1.1, 100)
    plt.plot(x_line, sigma_imp * x_line, 'k--', label=fr'Teoria ($\sigma_{{imp}}$ = {sigma_imp:.4f})')
    plt.plot(inv_r_arr, dp_arr, 'ro', markersize=8, label='Lattice Boltzmann (Numérico)')
    plt.plot(x_line, sigma_rec * x_line, 'b-', alpha=0.5, label=fr'Regressão Num ($\sigma_{{num}}$ = {sigma_rec:.4f})')

    plt.title(f"Validação Young-Laplace - Erro Relativo: {erro_relativo:.4f}%")
    plt.xlabel(r"Curvatura Analítica $1/R_0$ [lu$^{-1}$]")
    plt.ylabel(r"Diferencial de Pressão $\Delta p$ [lu]")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("validacao_laplace_acumulada.png", dpi=300)
    print(
        f"\nRecuperação da Tensão Superficial (σ):\nImposta: {sigma_imp:.5f} | Recuperada: {sigma_rec:.5f} | Erro: {erro_relativo:.4f}%")


if __name__ == "__main__":
    process_laplace_results()