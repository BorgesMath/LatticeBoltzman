# resultado_curvatura.py
"""
Calcula a curvatura da interface (phi=0) a partir de um arquivo .vtr exportado
pelo post_process.export_fields_vtk.

Uso:
    python resultado_curvatura.py <caminho_para_arquivo.vtr>

Requer:
    pip install vtk numpy
"""
import sys
import os
import numpy as np
import vtk
from vtk.util import numpy_support


def load_phi_from_vtr(vtr_path):
    """Lê o campo 'fase_phi' de um .vtr e devolve um array 2D (ny, nx)."""
    reader = vtk.vtkXMLRectilinearGridReader()
    reader.SetFileName(vtr_path)
    reader.Update()

    grid = reader.GetOutput()
    cell_data = grid.GetCellData()

    phi_array = cell_data.GetArray("fase_phi")
    if phi_array is None:
        nomes = [cell_data.GetArrayName(i) for i in range(cell_data.GetNumberOfArrays())]
        raise ValueError(
            f"Campo 'fase_phi' não encontrado em {vtr_path}. "
            f"Arrays disponíveis: {nomes}"
        )

    phi_flat = numpy_support.vtk_to_numpy(phi_array)

    # GetDimensions() retorna o número de PONTOS; o número de CÉLULAS é pontos-1
    dims_pts = grid.GetDimensions()  # (nx+1, ny+1, nz+1)
    nx = dims_pts[0] - 1
    ny = dims_pts[1] - 1

    # VTK armazena cell data com i (x) variando mais rápido.
    # Reshape como (nz=1, ny, nx) em ordem C devolve a convenção (y, x) original.
    phi_2d = phi_flat.reshape((1, ny, nx))[0, :, :]

    return phi_2d


def compute_curvature_field(phi):
    """
    Curvatura κ do level-set φ=0:
        κ = (φ_x²·φ_yy - 2·φ_x·φ_y·φ_xy + φ_y²·φ_xx) / |∇φ|³
    Mesma fórmula usada em post_process.compute_interface_curvature.
    """
    dy, dx = np.gradient(phi)         # dy = ∂φ/∂y, dx = ∂φ/∂x
    d2y, dy_dx = np.gradient(dy)      # d2y = ∂²φ/∂y², dy_dx = ∂²φ/∂y∂x
    dx_dy, d2x = np.gradient(dx)      # d2x = ∂²φ/∂x²

    num = (dx ** 2) * d2y + (dy ** 2) * d2x - 2.0 * dx * dy * dx_dy
    den = (dx ** 2 + dy ** 2 + 1e-8) ** 1.5
    kappa = num / den
    return kappa


def main():
    if len(sys.argv) < 2:
        print("Uso: python resultado_curvatura.py <caminho_para_arquivo.vtr>")
        sys.exit(1)

    vtr_path = sys.argv[1]
    if not os.path.isfile(vtr_path):
        print(f"[ERRO] Arquivo não encontrado: {vtr_path}")
        sys.exit(1)

    phi = load_phi_from_vtr(vtr_path)
    kappa = compute_curvature_field(phi)

    interface_mask = np.abs(phi) < 0.1
    n_interface = int(np.sum(interface_mask))

    print(f"\nArquivo            : {vtr_path}")
    print(f"Dimensões (ny, nx) : {phi.shape}")
    print(f"phi: min={phi.min():.4f}, max={phi.max():.4f}")

    if n_interface == 0:
        print("\n[AVISO] Nenhum ponto na interface (|phi| < 0.1) encontrado.")
        sys.exit(0)

    kappa_int = kappa[interface_mask]
    abs_kappa_int = np.abs(kappa_int)
    media_abs = float(np.mean(abs_kappa_int))

    print(f"\n--- Estatísticas de curvatura na interface (|phi| < 0.1) ---")
    print(f"Pontos na interface : {n_interface}")
    print(f"|κ| médio           : {media_abs:.6e}")
    print(f"|κ| máximo          : {float(np.max(abs_kappa_int)):.6e}")
    print(f"|κ| mediano         : {float(np.median(abs_kappa_int)):.6e}")
    print(f"|κ| desvio padrão   : {float(np.std(abs_kappa_int)):.6e}")
    print(f"κ médio (com sinal) : {float(np.mean(kappa_int)):.6e}")

    if media_abs > 1e-12:
        raio_medio = 1.0 / media_abs
        print(f"Raio médio (1/|κ̄|) : {raio_medio:.4f} células")
    else:
        print("Raio médio          : interface efetivamente plana (κ ≈ 0)")


if __name__ == "__main__":
    main()