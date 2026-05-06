# post_process/valida_lsa.py
"""
Compara o crescimento numérico da amplitude interfacial com a previsão analítica
da relação de dispersão adimensional (Eq. 9 — instabilidade de Saffman-Taylor magnética
em meio poroso).

Convenções de normalização:
    L_ref = NY          (altura do domínio, define a periodicidade)
    α     = 2π·m        (nº de onda adimensional do modo m)
    ζ     = s·L_ref/U   (taxa adimensional)  →  s = ζ·U_inlet/NY  [1/timestep]
    Da    = K₀ / NY²
    Ca    = ν_in · U / σ
    Ca_m  = χ · H₀² · NY / σ   (Bond magnético — igual ao relatorio_execucao.json)
    Λ     = χ / (2 + χ)         (contraste de suscetibilidade, fluido-2 não-magnético)
    H0n²  = cos²(θ),  H0t² = sin²(θ)   com θ = H_ANGLE em rad

Uso:
    python post_process/valida_lsa.py <diretorio_caso> [--t0 T0] [--t1 T1]

    --t0, --t1 : limites do janela de ajuste exponencial (em timesteps).
                 Por padrão usa 5 % – 45 % do intervalo total.

Exemplo:
    python post_process/valida_lsa.py 00_Testte_de_velocidade_d04mes05-h10_min30
    python post_process/valida_lsa.py 00_Testte_de_velocidade_d04mes05-h10_min30 --t0 50 --t1 400
"""

import sys
import os
import re
import glob
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

try:
    import vtk
    from vtk.util import numpy_support
    _HAS_VTK = True
except ImportError:
    _HAS_VTK = False


# ─────────────────────────────────────────────────────────────────────
# Estilo acadêmico (idêntico ao resultado_curvatura_temporal.py)
# ─────────────────────────────────────────────────────────────────────
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


# ═══════════════════════════════════════════════════════════════════
# 1.  RELAÇÃO DE DISPERSÃO  (Eq. 9)
# ═══════════════════════════════════════════════════════════════════
def zeta_analitico(alpha, M, Bo, Ca_m, Lambda_m, H0n_sq, H0t_sq, Da, Ca):
    """
    Taxa de crescimento adimensional ζ(α) — LSA Darcy padrão (Eq. 9).
    """
    termo_viscoso = alpha * ((M - 1.0) / (M + 1.0))
    termo_gravitacional = Bo
    termo_capilar = -(alpha ** 2)
    termo_magnetico = Ca_m * Lambda_m * alpha * (H0n_sq - H0t_sq)

    multiplicador = (Da / (Ca * (1.0 + M))) * alpha

    return termo_viscoso + multiplicador * (
            termo_gravitacional + termo_capilar + termo_magnetico
    )

# ═══════════════════════════════════════════════════════════════════
# 2.  PARÂMETROS ADIMENSIONAIS  ←  relatorio_execucao.json
# ═══════════════════════════════════════════════════════════════════
def extrair_parametros(case_dir):
    """
    Lê relatorio_execucao.json e devolve dict com todos os grupos
    adimensionais necessários para a comparação LSA.
    """
    json_path = os.path.join(case_dir, "relatorio_execucao.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"relatorio_execucao.json não encontrado em '{case_dir}'.\n"
            "Execute a simulação primeiro (main.py) para gerar o relatório."
        )

    with open(json_path, encoding='utf-8') as f:
        log = json.load(f)

    p = log["parametros_contorno"]

    ny      = float(p["NY"])
    tau_in  = p["TAU_IN"]
    tau_out = p["TAU_OUT"]
    nu_in   = (tau_in  - 0.5) / 3.0
    nu_out  = (tau_out - 0.5) / 3.0
    sigma   = p["SIGMA"]
    k0      = p["K_0"]
    u_in    = p["U_INLET"]
    h0      = p.get("H0", 0.0)
    chi_max = p.get("CHI_MAX", 0.0)
    h_angle = p.get("H_ANGLE", 0.0)   # graus
    mode_m  = int(p["mode_m"])

    if mode_m <= 0:
        raise ValueError(
            f"mode_m = {mode_m} — a análise LSA requer mode_m ≥ 1 "
            "(perturbação interfacial periódica)."
        )

    # ── Grupos adimensionais ──────────────────────────────────────
    L_ref = ny                             # comprimento de referência = NY

    M    = nu_out / nu_in                  # razão de viscosidades  μ_out / μ_in
    Da   = k0 / (L_ref ** 2)              # Darcy
    Ca   = (nu_in * u_in) / sigma         # capilaridade
    Bo   = 0.0                            # sem gravidade nesta formulação

    # Componentes angulares do campo (normalizadas: H0n² + H0t² = 1)
    theta  = np.radians(h_angle)
    H0n_sq = np.cos(theta) ** 2
    H0t_sq = np.sin(theta) ** 2

    # Bond magnético: Ca_m = χ · H₀² · L_ref / σ  (= Bo_mag do relatorio_execucao)
    Ca_m     = (chi_max * h0 ** 2 * L_ref) / sigma if sigma > 0 else 0.0
    # Contraste de suscetibilidade para fluido 2 não-magnético (χ₂ = 0)
    Lambda_m = chi_max / (2.0 + chi_max) if chi_max > 0 else 0.0

    # Número de onda adimensional do modo m com L_ref = NY:  α = k · NY = 2π · m
    alpha_sim = 2.0 * np.pi * mode_m

    return dict(
        M=M, Da=Da, Ca=Ca, Bo=Bo,
        Ca_m=Ca_m, Lambda_m=Lambda_m,
        H0n_sq=H0n_sq, H0t_sq=H0t_sq,
        alpha_sim=alpha_sim,
        L_ref=L_ref, U_ref=u_in,
        mode_m=mode_m, ny=int(ny),
        u_in=u_in, nu_in=nu_in, sigma=sigma,
        h0=h0, chi_max=chi_max, h_angle=h_angle, k0=k0,
    )


# ═══════════════════════════════════════════════════════════════════
# 3.  AMPLITUDE NUMÉRICA  ←  cache .npz  ou  VTKs
# ═══════════════════════════════════════════════════════════════════
def _phi_de_vtr(vtr_path):
    reader = vtk.vtkXMLRectilinearGridReader()
    reader.SetFileName(vtr_path)
    reader.Update()
    grid = reader.GetOutput()
    arr  = grid.GetCellData().GetArray("fase_phi")
    if arr is None:
        raise ValueError(f"'fase_phi' ausente em {vtr_path}")
    pts = grid.GetDimensions()
    nx, ny = pts[0] - 1, pts[1] - 1
    return numpy_support.vtk_to_numpy(arr).reshape((ny, nx))


def _amplitude(phi, mode_m):
    """
    Calcula a amplitude do modo-m via projeção de Fourier da posição da interface phi=0.
    Mais robusto que max-min para ondas com ruído ou modos mistos.
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

    phase = np.exp(-2j * np.pi * mode_m * linhas / ny)
    A_m = (2.0 / ny) * abs(np.sum(x_exact * phase))
    return float(A_m)


def _ts_de_nome(fname):
    m = re.search(r'dados_macro_(\d+)\.vtr$', os.path.basename(fname))
    return int(m.group(1)) if m else -1


def carregar_amplitude(case_dir, mode_m, max_snaps=80):
    """
    Retorna (t, A) arrays.  Tenta o .npz de cache primeiro; se ausente, lê VTKs.
    """
    npz = os.path.join(case_dir, "curvatura_temporal.npz")
    if os.path.exists(npz):
        data = np.load(npz)
        print(f"[info] Amplitude carregada de cache: {npz}")
        return data['t'].astype(float), data['amplitude'].astype(float)

    if not _HAS_VTK:
        raise ImportError(
            "Módulo 'vtk' não disponível.  Gere primeiro o curvatura_temporal.npz "
            "com resultado_curvatura_temporal.py, ou instale vtk."
        )

    vtk_dir = os.path.join(case_dir, "vtk")
    vtrs = sorted(
        glob.glob(os.path.join(vtk_dir, "dados_macro_*.vtr")),
        key=_ts_de_nome
    )
    if not vtrs:
        raise FileNotFoundError(f"Nenhum .vtr encontrado em {vtk_dir}")

    if len(vtrs) > max_snaps:
        idx  = np.linspace(0, len(vtrs) - 1, max_snaps, dtype=int)
        vtrs = [vtrs[i] for i in idx]

    times, amps = [], []
    for i, fpath in enumerate(vtrs):
        ts  = _ts_de_nome(fpath)
        phi = _phi_de_vtr(fpath)
        A   = _amplitude(phi, mode_m)
        times.append(ts); amps.append(A)
        print(f"  [{i+1:3d}/{len(vtrs)}]  t={ts:6d}   A = {A:.3f} l.u.")

    return np.array(times, dtype=float), np.array(amps, dtype=float)


# ═══════════════════════════════════════════════════════════════════
# 4.  AJUSTE EXPONENCIAL  A(t) = A₀ · exp(s · t)
# ═══════════════════════════════════════════════════════════════════
def ajuste_exponencial(t, A, t0_user=None, t1_user=None):
    """
    Regressão linear em log-espaço na janela de regime linear.

    Se t0_user / t1_user são None, usa [5%, 45%] do intervalo total.
    Retorna (s_fit, A0_fit, t_janela, A_janela).
    """
    t_span = t[-1] - t[0]
    t0 = t0_user if t0_user is not None else t[0] + 0.05 * t_span
    t1 = t1_user if t1_user is not None else t[0] + 0.45 * t_span

    mask = (t >= t0) & (t <= t1) & (A > 0)
    if mask.sum() < 3:
        print("[aviso] Janela de ajuste com < 3 pontos; usando todos os pontos A > 0.")
        mask = A > 0
    if mask.sum() < 2:
        raise ValueError("Pontos insuficientes para o ajuste exponencial (A > 0).")

    t_w, A_w = t[mask], A[mask]
    coeffs = np.polyfit(t_w, np.log(A_w), 1)
    return float(coeffs[0]), float(np.exp(coeffs[1])), t_w, A_w


# ═══════════════════════════════════════════════════════════════════
# 5.  RELATÓRIO NO TERMINAL
# ═══════════════════════════════════════════════════════════════════
def _relatorio(params, s_num, s_ana):
    zeta_num = s_num * params['L_ref'] / params['U_ref']
    zeta_ana = zeta_analitico(
        params['alpha_sim'],
        params['M'], params['Bo'], params['Ca_m'], params['Lambda_m'],
        params['H0n_sq'], params['H0t_sq'], params['Da'], params['Ca']
    )
    erro_pct = abs(s_num - s_ana) / (abs(s_ana) + 1e-30) * 100.0

    sep = "─" * 58
    print(f"\n{sep}")
    print("   COMPARAÇÃO  LBM  ↔  LSA  (Assintótico Linear)")
    print(sep)
    print(f"   Modo simulado       m   = {params['mode_m']}")
    print(f"   α = 2π·m           α   = {params['alpha_sim']:.4f}")
    print(f"   M  (visc. ratio)   M   = {params['M']:.6f}")
    print(f"   Da (Darcy)         Da  = {params['Da']:.4e}")
    print(f"   Ca (capilar)       Ca  = {params['Ca']:.4e}")
    print(f"   Ca_m (Bond mag.)   Ca_m= {params['Ca_m']:.4e}")
    print(f"   Λ  (contraste χ)   Λ   = {params['Lambda_m']:.4f}")
    print(f"   H0n² (normal)           = {params['H0n_sq']:.4f}")
    print(f"   H0t² (tangencial)       = {params['H0t_sq']:.4f}")
    print(sep)
    print(f"   Taxa dimensional  [1/timestep]:")
    print(f"     Numérica  (ajuste)   s_num = {s_num:+.6e}")
    print(f"     Analítica (Eq. 9)   s_ana = {s_ana:+.6e}")
    print(f"     Erro relativo              = {erro_pct:.2f} %")
    print(sep)
    print(f"   Taxa adimensional  ζ:")
    print(f"     Numérica            ζ_num = {zeta_num:+.4f}")
    print(f"     Analítica           ζ_ana = {zeta_ana:+.4f}")
    print(sep)
    return zeta_num, zeta_ana


# ═══════════════════════════════════════════════════════════════════
# 6.  FIGURA
# ═══════════════════════════════════════════════════════════════════
def plotar(t, A, s_num, A0_num, s_ana, params, t_janela, case_dir):
    zeta_num = s_num * params['L_ref'] / params['U_ref']
    zeta_ana = zeta_analitico(
        params['alpha_sim'],
        params['M'], params['Bo'], params['Ca_m'], params['Lambda_m'],
        params['H0n_sq'], params['H0t_sq'], params['Da'], params['Ca']
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # ── Painel esquerdo: A(t) ─────────────────────────────────────
    t_fine = np.linspace(t[0], t[-1], 800)
    # Exponencial numérica (ajuste)
    A_fit  = A0_num * np.exp(s_num * t_fine)
    # Exponencial analítica ancorada no primeiro ponto válido
    A0_ana = A[A > 0][0] if np.any(A > 0) else 1.0
    t0_ana = t[A > 0][0] if np.any(A > 0) else t[0]
    A_ana  = A0_ana * np.exp(s_ana * (t_fine - t0_ana))

    ax1.semilogy(t, A, 'o', ms=4.5, color='#555', zorder=3,
                 markerfacecolor='white', markeredgewidth=1.2,
                 label="Numérico (LBM)")
    ax1.semilogy(t_fine, A_fit, '-', color='#1a5276', lw=1.8,
                 label=fr"Ajuste exp.  $s_{{num}}={s_num:.3e}$ ts$^{{-1}}$")
    ax1.semilogy(t_fine, A_ana, '--', color='#922b21', lw=1.8,
                 label=fr"Analítico     $s_{{ana}}={s_ana:.3e}$ ts$^{{-1}}$")

    # Sombreia a janela de ajuste
    ax1.axvspan(t_janela[0], t_janela[-1], alpha=0.08, color='#1a5276',
                label="Janela de ajuste")

    ax1.set_xlabel(r"Passo de tempo $t$")
    ax1.set_ylabel(r"Amplitude interfacial $A$ (l.u.)")
    ax1.set_title(r"Crescimento da Perturbação: $A(t) \propto e^{s\,t}$")
    ax1.legend(frameon=True, fancybox=False, edgecolor='black', fontsize=9,
               loc='upper left')
    ax1.tick_params(direction='in', top=True, right=True)

    # ── Painel direito: curva de dispersão ζ(α) ──────────────────
    alpha_max   = max(3.5 * params['alpha_sim'], 15.0)
    alpha_range = np.linspace(1e-4, alpha_max, 1000)
    zeta_curve  = zeta_analitico(
        alpha_range,
        params['M'], params['Bo'], params['Ca_m'], params['Lambda_m'],
        params['H0n_sq'], params['H0t_sq'], params['Da'], params['Ca']
    )

    ax2.plot(alpha_range, zeta_curve, '-', color='#1a5276', lw=1.8,
             label=r"$\zeta(\alpha)$ analítico  (Eq. 9)")
    ax2.axhline(0, color='black', lw=0.9, ls='--', alpha=0.6)
    ax2.axvline(params['alpha_sim'], color='#888', ls=':', lw=1.0)

    # Ponto analítico do modo m
    ax2.scatter([params['alpha_sim']], [zeta_ana], zorder=6, s=70,
                color='#922b21',
                label=(fr"$\zeta_{{ana}}(\alpha_m) = {zeta_ana:.3f}$"
                       fr"   ($m={params['mode_m']},\ \alpha={params['alpha_sim']:.2f}$)"))

    # Ponto numérico (ζ convertido de s_num)
    ax2.scatter([params['alpha_sim']], [zeta_num], zorder=6, s=70,
                marker='D', color='#1e8449',
                label=fr"$\zeta_{{num}} = {zeta_num:.3f}$  (LBM ajustado)")

    ax2.set_xlabel(r"Número de onda adimensional $\alpha = 2\pi m$")
    ax2.set_ylabel(r"Taxa de crescimento $\zeta$")
    ax2.set_title(r"Relação de Dispersão — Eq. 9")
    ax2.legend(frameon=True, fancybox=False, edgecolor='black', fontsize=9)
    ax2.tick_params(direction='in', top=True, right=True)

    # Limites verticais: mostra a curva com margem
    finite_mask = np.isfinite(zeta_curve)
    if np.any(finite_mask):
        y_lo = min(0.0, np.percentile(zeta_curve[finite_mask], 2))
        y_hi = max(0.0, np.percentile(zeta_curve[finite_mask], 98))
        margin = 0.15 * (y_hi - y_lo) if y_hi != y_lo else 1.0
        ax2.set_ylim(y_lo - margin, y_hi + margin)

    fig.tight_layout()
    out_path = os.path.join(case_dir, "comparacao_lsa.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Figura salva em: {out_path}")


# ═══════════════════════════════════════════════════════════════════
# 7.  MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Compara crescimento LBM com relação de dispersão analítica (Eq. 9)."
    )
    parser.add_argument("case_dir", help="Diretório de saída da simulação.")
    parser.add_argument("--t0", type=float, default=None,
                        help="Início da janela de ajuste exponencial (timestep).")
    parser.add_argument("--t1", type=float, default=None,
                        help="Fim    da janela de ajuste exponencial (timestep).")
    args = parser.parse_args()

    case_dir = args.case_dir
    if not os.path.isdir(case_dir):
        print(f"[ERRO] Diretório inválido: {case_dir}")
        sys.exit(1)

    print(f"\n{'═'*58}")
    print(f"  valida_lsa.py  —  {case_dir}")
    print(f"{'═'*58}")

    params = extrair_parametros(case_dir)

    print(f"\nCarregando série temporal de amplitude...")
    t, A = carregar_amplitude(case_dir, params['mode_m'])

    print(f"\nAjustando exponencial no regime linear...")
    s_num, A0_num, t_janela, A_janela = ajuste_exponencial(t, A, args.t0, args.t1)

    # Taxa analítica dimensional  [1/timestep]
    zeta_ana = zeta_analitico(
        params['alpha_sim'],
        params['M'], params['Bo'], params['Ca_m'], params['Lambda_m'],
        params['H0n_sq'], params['H0t_sq'], params['Da'], params['Ca']
    )
    s_ana = zeta_ana * params['U_ref'] / params['L_ref']

    _relatorio(params, s_num, s_ana)
    plotar(t, A, s_num, A0_num, s_ana, params, t_janela, case_dir)


if __name__ == "__main__":
    main()
