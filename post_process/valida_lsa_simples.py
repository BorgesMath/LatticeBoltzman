# post_process/valida_lsa_simples.py
"""
Variante de valida_lsa.py que usa o metodo de amplitude MAIS SIMPLES possível:
a excursão pico-a-pico da posição da interface,

    A(t) = ½ ( max_y x*(y)  −  min_y x*(y) ),

em vez da projeção de Fourier do modo m. Tudo o mais (detecção sub-pixel da
interface φ=0, ajuste exponencial, platô de s_local, relação de dispersão
analítica Eq. 9 e métricas de erro) é reaproveitado de valida_lsa.py, de modo
que a ÚNICA diferença em relação ao metodo de Fourier é a métrica de amplitude.

Importante: NÃO usa o cache curvatura_temporal.npz (que contém a amplitude de
Fourier). Relê sempre os VTKs e subamostra para o mesmo número de instantes
(max_snaps=80, igual ao cache de Fourier) para uma comparação justa.

Uso:
    python post_process/valida_lsa_simples.py <diretorio_caso> [--t0 T0] [--t1 T1] [--no-auto]
"""

import sys
import os
import glob
import argparse
import numpy as np

# Reaproveita toda a física já validada (parâmetros, dispersão, ajuste, plot).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import valida_lsa as vl


# ═══════════════════════════════════════════════════════════════════
#  AMPLITUDE PICO-A-PICO  (metodo simples)
# ═══════════════════════════════════════════════════════════════════
def _amplitude_simples(phi):
    """
    Semi-amplitude pico-a-pico da interface φ=0.

    Usa a MESMA detecção sub-pixel de valida_lsa._amplitude (interpolação
    linear no cruzamento de sinal), mas em vez da projeção de Fourier retorna

        A = ½ ( max_y x*(y) − min_y x*(y) ).
    """
    ny, nx = phi.shape

    idx_right = np.argmax(phi < 0.0, axis=1)
    valid = idx_right > 0
    if not np.any(valid):
        return 0.0

    idx_r = idx_right[valid]
    idx_l = idx_r - 1
    linhas = np.arange(ny)[valid]

    phi_r = phi[linhas, idx_r]
    phi_l = phi[linhas, idx_l]
    x_exact = idx_l + phi_l / (phi_l - phi_r + 1e-15)

    return float(0.5 * (np.max(x_exact) - np.min(x_exact)))


def carregar_amplitude_simples(case_dir, max_snaps=80):
    """
    Lê os VTKs (ignora o cache de Fourier) e devolve (t, A) com a amplitude
    pico-a-pico. Subamostra para max_snaps instantes igualmente espaçados,
    reproduzindo a mesma malha temporal do cache de Fourier.
    """
    if not vl._HAS_VTK:
        raise ImportError("Módulo 'vtk' indisponível — necessário para reler os VTKs.")

    vtk_dir = os.path.join(case_dir, "vtk")
    vtrs = sorted(
        glob.glob(os.path.join(vtk_dir, "dados_macro_*.vtr")),
        key=vl._ts_de_nome
    )
    if not vtrs:
        raise FileNotFoundError(f"Nenhum .vtr encontrado em {vtk_dir}")

    if len(vtrs) > max_snaps:
        idx = np.linspace(0, len(vtrs) - 1, max_snaps, dtype=int)
        vtrs = [vtrs[i] for i in idx]

    times, amps = [], []
    for i, fpath in enumerate(vtrs):
        ts = vl._ts_de_nome(fpath)
        phi = vl._phi_de_vtr(fpath)
        A = _amplitude_simples(phi)
        times.append(ts); amps.append(A)
        print(f"  [{i+1:3d}/{len(vtrs)}]  t={ts:6d}   A_pp = {A:.3f} l.u.")

    return np.array(times, dtype=float), np.array(amps, dtype=float)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Validação LSA usando amplitude pico-a-pico (método simples)."
    )
    parser.add_argument("case_dir", help="Diretório de saída da simulação.")
    parser.add_argument("--t0", type=float, default=None,
                        help="Início da janela de ajuste exponencial (timestep).")
    parser.add_argument("--t1", type=float, default=None,
                        help="Fim    da janela de ajuste exponencial (timestep).")
    parser.add_argument("--no-auto", action="store_true",
                        help="Desativa a detecção automática do platô.")
    args = parser.parse_args()

    case_dir = args.case_dir
    if not os.path.isdir(case_dir):
        print(f"[ERRO] Diretório inválido: {case_dir}")
        sys.exit(1)

    print(f"\n{'═'*58}")
    print(f"  valida_lsa_simples.py  —  {case_dir}")
    print(f"  [método] amplitude PICO-A-PICO  A = ½(max−min)")
    print(f"{'═'*58}")

    params = vl.extrair_parametros(case_dir)

    print(f"\nRelendo VTKs e medindo amplitude pico-a-pico...")
    t, A = carregar_amplitude_simples(case_dir, max_snaps=80)

    print(f"\nAjustando exponencial no regime linear...")
    s_num, A0_num, t_janela, A_janela, info = vl.ajuste_exponencial(
        t, A, args.t0, args.t1, auto=not args.no_auto
    )

    zeta_ana = vl.zeta_analitico(
        params['alpha_sim'],
        params['M'], params['Bo'], params['Ca_m'], params['Lambda_m'],
        params['H0n_sq'], params['H0t_sq'], params['Da'], params['Ca']
    )
    s_ana = zeta_ana * params['U_ref'] / params['L_ref']

    vl._relatorio(params, s_num, s_ana, info)
    # Reutiliza o plot, mas salva com nome próprio.
    _plotar_simples(t, A, s_num, A0_num, s_ana, params, t_janela, case_dir, info)


def _plotar_simples(t, A, s_num, A0_num, s_ana, params, t_janela, case_dir, info):
    """Gera comparacao_lsa_simples.png reaproveitando vl.plotar via monkeypatch leve."""
    import matplotlib.pyplot as plt
    # vl.plotar salva sempre como comparacao_lsa.png; geramos e renomeamos.
    out_fourier = os.path.join(case_dir, "comparacao_lsa.png")
    backup = None
    if os.path.exists(out_fourier):
        backup = out_fourier + ".bak_tmp"
        os.replace(out_fourier, backup)

    vl.plotar(t, A, s_num, A0_num, s_ana, params, t_janela, case_dir, info)

    out_simples = os.path.join(case_dir, "comparacao_lsa_simples.png")
    if os.path.exists(out_fourier):
        os.replace(out_fourier, out_simples)
        print(f"[OK] Figura (método simples) salva em: {out_simples}")
    if backup is not None:
        os.replace(backup, out_fourier)


if __name__ == "__main__":
    main()
