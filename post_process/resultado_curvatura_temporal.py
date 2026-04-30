# resultado_curvatura_temporal.py
"""
Captura a curvatura média da interface ao longo do tempo a partir dos arquivos
.vtr exportados pelo post_process. Seleciona 20 snapshots uniformemente
espaçados no tempo (ou todos se houver menos de 20) e plota em estilo acadêmico.

Uso:
    python resultado_curvatura_temporal.py <diretorio_com_vtrs>

Exemplo:
    python resultado_curvatura_temporal.py 02_Instavel_Magnetismo_Normal_d12mes04-h15_min30/vtk
"""
import sys
import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import vtk
from vtk.util import numpy_support


# Estilo acadêmico (mesmo do valida_lsa.py)
plt.style.use('default')
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.6,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.6,
})


def load_phi_from_vtr(vtr_path):
    """Lê o campo 'fase_phi' de um .vtr e devolve um array 2D (ny, nx)."""
    reader = vtk.vtkXMLRectilinearGridReader()
    reader.SetFileName(vtr_path)
    reader.Update()
    grid = reader.GetOutput()
    phi_array = grid.GetCellData().GetArray("fase_phi")
    if phi_array is None:
        raise ValueError(f"'fase_phi' ausente em {vtr_path}")

    phi_flat = numpy_support.vtk_to_numpy(phi_array)
    dims_pts = grid.GetDimensions()
    nx = dims_pts[0] - 1
    ny = dims_pts[1] - 1
    return phi_flat.reshape((1, ny, nx))[0, :, :]


def compute_mean_abs_curvature(phi):
    """
    Curvatura κ do level-set φ=0:
        κ = (φ_x²·φ_yy + φ_y²·φ_xx - 2·φ_x·φ_y·φ_xy) / |∇φ|³
    Retorna |κ| médio na faixa interfacial |φ| < 0.1.
    """
    dy, dx = np.gradient(phi)
    d2y, dy_dx = np.gradient(dy)
    dx_dy, d2x = np.gradient(dx)

    num = (dx ** 2) * d2y + (dy ** 2) * d2x - 2.0 * dx * dy * dx_dy
    den = (dx ** 2 + dy ** 2 + 1e-8) ** 1.5
    kappa = num / den

    mask = np.abs(phi) < 0.1
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(kappa[mask])))


def compute_interface_amplitude(phi):
    """
    Amplitude da perturbação da interface: A = (X_max - X_min) / 2
    onde X é a posição x do primeiro ponto com φ < 0 em cada linha y.
    """
    interface_x = np.argmax(phi < 0.0, axis=1).astype(float)
    return float((np.max(interface_x) - np.min(interface_x)) / 2.0)


def extract_time_step(filename):
    """Extrai o passo t do padrão 'dados_macro_NNNNN.vtr'."""
    m = re.search(r'dados_macro_(\d+)\.vtr$', os.path.basename(filename))
    return int(m.group(1)) if m else -1


def main():
    if len(sys.argv) < 2:
        print("Uso: python resultado_curvatura_temporal.py <diretorio_com_vtrs>")
        sys.exit(1)

    vtk_dir = sys.argv[1]
    if not os.path.isdir(vtk_dir):
        print(f"[ERRO] Diretório inválido: {vtk_dir}")
        sys.exit(1)

    vtr_files = sorted(
        glob.glob(os.path.join(vtk_dir, "dados_macro_*.vtr")),
        key=extract_time_step
    )
    if len(vtr_files) == 0:
        print(f"[ERRO] Nenhum arquivo 'dados_macro_*.vtr' em {vtk_dir}")
        sys.exit(1)

    # Seleciona 20 snapshots uniformemente espaçados (ou todos, se menor)
    N_TARGET = 20
    if len(vtr_files) > N_TARGET:
        idx = np.linspace(0, len(vtr_files) - 1, N_TARGET, dtype=int)
        vtr_files = [vtr_files[i] for i in idx]

    print(f"Processando {len(vtr_files)} snapshots de {vtk_dir}\n")

    times = []
    curvs = []
    amps = []
    for i, f in enumerate(vtr_files):
        t = extract_time_step(f)
        phi = load_phi_from_vtr(f)
        k = compute_mean_abs_curvature(phi)
        a = compute_interface_amplitude(phi)
        times.append(t)
        curvs.append(k)
        amps.append(a)
        print(f"  [{i + 1:2d}/{len(vtr_files)}] t={t:6d}  |kappa_mean| = {k:.6e}  amplitude = {a:.2f}")

    times = np.array(times)
    curvs = np.array(curvs)
    amps = np.array(amps)

    # Identifica o caso pelo nome do diretório-pai (acima de /vtk)
    parent = os.path.basename(os.path.dirname(os.path.abspath(vtk_dir)))
    out_path = os.path.join(os.path.dirname(os.path.abspath(vtk_dir)),
                            "curvatura_temporal.png")

    # ---------- Plot acadêmico (curvatura + amplitude, subplots) ----------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 7.0), sharex=True)

    ax1.plot(times, curvs, marker='o', markersize=5, color='#9b1d20',
             markerfacecolor='white', markeredgecolor='#9b1d20',
             markeredgewidth=1.4, label=r"$\overline{|\kappa|}(t)$")
    ax1.set_ylabel(r"Mean absolute curvature  $\overline{|\kappa|}$")
    pos = curvs[curvs > 0]
    if pos.size > 0 and (pos.max() / pos.min() > 100):
        ax1.set_yscale('log')
    ax1.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
    ax1.tick_params(direction='in', top=True, right=True)

    ax2.plot(times, amps, marker='s', markersize=5, color='#1a5276',
             markerfacecolor='white', markeredgecolor='#1a5276',
             markeredgewidth=1.4, label=r"$A(t)$")
    ax2.set_xlabel(r"Time step $t$ (lattice units)")
    ax2.set_ylabel(r"Interface amplitude  $A$ (l.u.)")
    ax2.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
    ax2.tick_params(direction='in', top=True, right=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\n[OK] Gráfico salvo em: {out_path}")

    # Salva também em .npz para reuso (comparação entre casos)
    npz_path = os.path.join(os.path.dirname(os.path.abspath(vtk_dir)),
                            "curvatura_temporal.npz")
    np.savez(npz_path, t=times, kappa_mean_abs=curvs, amplitude=amps)
    print(f"[OK] Dados salvos em : {npz_path}")


if __name__ == "__main__":
    main()