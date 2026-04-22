import os
import glob
import json
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

# Diretório base (raiz do projeto onde estão as pastas de resultado)
BASE_DIR = os.getcwd()


def process_all_results():
    # Encontra todos os relatórios gerados
    report_files = glob.glob(os.path.join(BASE_DIR, "*", "relatorio_execucao.json"))

    k_imposed_list = []
    k_recovered_list = []

    print(f"Encontradas {len(report_files)} simulações com relatórios.")

    for report_path in report_files:
        folder = os.path.dirname(report_path)

        # Ler os parâmetros do JSON
        with open(report_path, 'r', encoding='utf-8') as f:
            log_data = json.load(f)

        params = log_data.get("parametros_contorno", {})
        k_imp = params.get("K_0")
        tau_in = params.get("TAU_IN")
        ch_substeps = params.get("CH_SUBSTEPS", 1)
        mode_m = params.get("mode_m", -1)

        # Filtra estritamente os casos unifásicos de Darcy (sem Cahn-Hilliard)
        if ch_substeps > 0 or mode_m != 0:
            continue

        if k_imp is None or tau_in is None:
            continue

        nu = (tau_in - 0.5) / 3.0

        # Procura os arquivos VTK (.vtr) dentro do diretório gerado
        vtk_files = glob.glob(os.path.join(folder, "vtk", "*.vtr"))
        if not vtk_files:
            continue

        # Seleciona o último arquivo gerado (estado estacionário)
        latest_vtk = max(vtk_files, key=os.path.getmtime)

        try:
            mesh = pv.read(latest_vtk)

            # Como a exportação usa pyevtk em grade regular, dimensions retorna NÓS.
            # O número de CÉLULAS é NÓS - 1.
            NX_nodes, NY_nodes, _ = mesh.dimensions
            NX, NY = NX_nodes - 1, NY_nodes - 1

            # Extrai os tensores estruturados gravados no cellData
            ux_flat = mesh.cell_data['velocidade'][:, 0]
            rho_flat = mesh.cell_data['densidade_rho']

            # Reversão da transposição Fortran-like do pyevtk no post_process.py
            # A operação .reshape((NX, NY), order='F').T retorna exatamente a matriz LBM (NY, NX)
            ux_2d = ux_flat.reshape((NX, NY), order='F').T
            rho_2d = rho_flat.reshape((NX, NY), order='F').T

        except Exception as e:
            print(f"Pasta {os.path.basename(folder)}: Erro ao processar malha VTK ({e}).")
            continue

        # Equação de Estado Isotérmica
        p_2d = rho_2d / 3.0

        y_center = NY // 2

        # Descartar 25% de cada extremidade (Inlet/Outlet) para evitar efeitos acústicos de borda
        bulk_start, bulk_end = int(0.25 * NX), int(0.75 * NX)
        x_bulk = np.arange(NX)[bulk_start:bulk_end]

        p_bulk = p_2d[y_center, bulk_start:bulk_end]
        u_avg_bulk = np.mean(ux_2d[y_center, bulk_start:bulk_end])
        rho_avg_bulk = np.mean(rho_2d[y_center, bulk_start:bulk_end])

        # Extração do Gradiente Macroscópico via Regressão Linear
        coeffs = np.polyfit(x_bulk, p_bulk, 1)
        dp_dx = coeffs[0]

        # Evita divisão por zero
        if abs(dp_dx) > 1e-15:
            # Reconstrução LBM da permeabilidade de Darcy: K = (nu * rho * u) / (-dp/dx)
            k_rec = (nu * rho_avg_bulk * u_avg_bulk) / (-dp_dx)
            k_imposed_list.append(k_imp)
            k_recovered_list.append(k_rec)
            print(
                f"Pasta {os.path.basename(folder)}: K_imp={k_imp:6.2f} | K_rec={k_rec:6.4f} | Erro={abs(k_imp - k_rec) / k_imp * 100:5.2f}%")
        else:
            print(f"Pasta {os.path.basename(folder)}: Gradiente de pressão nulo detectado.")

    if not k_imposed_list:
        print("\nNenhum caso unifásico válido da Lei de Darcy foi encontrado para gerar os gráficos.")
        return

    # Ordenação topológica para plotagem do gráfico
    idx_sort = np.argsort(k_imposed_list)
    k_imp_arr = np.array(k_imposed_list)[idx_sort]
    k_rec_arr = np.array(k_recovered_list)[idx_sort]
    errors = np.abs(k_imp_arr - k_rec_arr) / k_imp_arr * 100

    # Configuração de Estilo do Plot (Ortodoxo, Acadêmico)
    plt.style.use('default')
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Curva 1: Recuperação Analítica (Escala Log-Log)
    ax1.loglog(k_imp_arr, k_imp_arr, 'k--', label='Ideal (Paridade)')
    ax1.loglog(k_imp_arr, k_rec_arr, 'ro', markersize=8, label='LBM Numérico')
    ax1.set_xlabel(r'Permeabilidade Imposta $K_0$ [lu]')
    ax1.set_ylabel(r'Permeabilidade Recuperada $K_{num}$ [lu]')
    ax1.legend(frameon=False)
    ax1.set_title('Validação de Darcy - Brinkman/Forchheimer')
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    # Curva 2: Erro Relativo (Escala Semilog-x)
    ax2.plot(k_imp_arr, errors, 'ks-', markersize=6)
    ax2.set_xlabel(r'Permeabilidade Imposta $K_0$ [lu]')
    ax2.set_ylabel('Erro Relativo Computacional (%)')
    ax2.set_title('Análise de Erro L2 Residual')
    ax2.set_xscale('log')
    ax2.grid(True, which="both", ls=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig("validacao_darcy_lote.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    process_all_results()