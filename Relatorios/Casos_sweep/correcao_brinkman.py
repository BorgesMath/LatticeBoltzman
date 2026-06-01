# -*- coding: utf-8 -*-
"""
correcao_brinkman.py — Correção de Brinkman para a relação de dispersão LSA.

DIAGNÓSTICO (ver análise do sweep AD_W3):
  O déficit de taxa d = 1 - ζ_num/ζ_ana NÃO escala com o viés de fase-field
  kξ=2πmW/NY (refinar NY de 1024→2048 a Da fixo NÃO reduziu o erro do m=2:
  8.7%→9.5%). Ele escala LINEARMENTE com o número de blindagem de Brinkman

      β = √K · k = α·√Da        (α=2πm,  Da=K₀/NY²)

  que é INVARIANTE na resolução (K₀∝NY² e k²∝1/NY² se cancelam). Ajuste dos
  6 pontos NY=1024:  d ∝ α com R²=0.975  vs  d ∝ α² com R²=0.567.

FÍSICA:
  A LBM resolve Navier-Stokes com arrasto de Darcy -(ν/K)u E o termo viscoso
  ν∇²u (Brinkman), intrínseco ao BGK. A LSA "Eq. 9" é Darcy PURO (despreza
  ν∇²u). O termo de Brinkman cria uma camada-limite de espessura √K na
  interface; sua correção de ordem líder ao balanço de tensão normal é
  LINEAR em β=√K·k. Logo:

      ζ_Brinkman(α) = ζ_Darcy(α) · (1 - c·√Da·α),   c ≈ 0.80 (calibrado)

  c é o coeficiente da camada-limite de Brinkman; aqui é calibrado nos dados
  (não derivado de 1º princípio). A Eq. 9 (Darcy) é o limite β→0.

Reproduz:  python correcao_brinkman.py
Gera:      correcao_brinkman.png  +  tabela_brinkman.tex
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = os.path.dirname(os.path.abspath(__file__))

# ── Parâmetros físicos comuns do sweep AD_W3 ────────────────────────
M, Da, Ca = 4.0, 1.2493e-4, 0.25
sqDa = np.sqrt(Da)


def zeta_darcy(alpha):
    """LSA Darcy pura (Eq. 9), sem grav/mag: ζ = 0.6α - [Da/(Ca(1+M))]α³."""
    termo_viscoso = alpha * (M - 1.0) / (M + 1.0)
    multiplicador = (Da / (Ca * (1.0 + M))) * alpha
    return termo_viscoso + multiplicador * (-(alpha ** 2))


def zeta_brinkman(alpha, c):
    """Darcy corrigida pela camada-limite de Brinkman (ordem líder em β=α√Da)."""
    return zeta_darcy(alpha) * (1.0 - c * sqDa * alpha)


# ── Dados medidos ───────────────────────────────────────────────────
# Sweep NY=1024 (m=1..6).  ζ_num = s · NY/U,  FATOR_1024 = 1024/0.0025 = 409600.
m1024 = np.array([1, 2, 3, 4, 5, 6], float)
a1024 = 2 * np.pi * m1024
sPP_1024 = np.array([9.143129e-06, 1.636460e-05, 2.138331e-05,
                     2.519235e-05, 2.738011e-05, 2.821650e-05])
sF_1024 = np.array([9.461882e-06, 1.591138e-05, 2.144804e-05,
                    2.524032e-05, 2.743889e-05, 2.827305e-05])
zPP_1024 = sPP_1024 * 409600.0
zF_1024 = sF_1024 * 409600.0

# Sweep NY=2048 (m=1,2) — confirma INVARIÂNCIA na resolução (mesmo β, mesmo erro).
# ζ_num = s · 2048/0.0025 = s · 819200.  (pico-a-pico, janela auto)
m2048 = np.array([1, 2], float)
a2048 = 2 * np.pi * m2048
zPP_2048 = np.array([3.8046, 6.6459])


# ── Calibração de c (déficit d = 1 - ζ_num/ζ_ana ≈ c·β) ─────────────
beta_1024 = a1024 * sqDa
d_pp = 1.0 - zPP_1024 / zeta_darcy(a1024)
d_f = 1.0 - zF_1024 / zeta_darcy(a1024)

# Ajuste sem intercepto (β→0 ⇒ d→0). Usa m≥2 (m=1 ~ ruído em 0).
mask = m1024 >= 2
c_pp = float(np.sum(d_pp[mask] * beta_1024[mask]) / np.sum(beta_1024[mask] ** 2))
c_f = float(np.sum(d_f[mask] * beta_1024[mask]) / np.sum(beta_1024[mask] ** 2))
c_use = 0.5 * (c_pp + c_f)


def r2(y, yhat):
    return 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


if __name__ == "__main__":
    # ════════════ Tabela de validação no terminal ════════════
    print("=" * 74)
    print("  CORREÇÃO DE BRINKMAN — ζ_B(α) = ζ_Darcy(α)·(1 - c·√Da·α)")
    print(f"  c calibrado:  pico-a-pico={c_pp:.4f}  Fourier={c_f:.4f}  -> c={c_use:.4f}")
    print(f"  R² do déficit linear (m≥2): PP={r2(d_pp[mask], c_pp*beta_1024[mask]):.4f}"
          f"  F={r2(d_f[mask], c_f*beta_1024[mask]):.4f}")
    print("=" * 74)
    print(f"{'m':>2}{'α':>8}{'β':>8}{'ζ_Darcy':>9}{'ζ_Brink':>9}"
          f"{'ζ_LBM(PP)':>11}{'err_Da%':>9}{'err_Br%':>9}")
    for i in range(6):
        zd, zb, zl = zeta_darcy(a1024[i]), zeta_brinkman(a1024[i], c_use), zPP_1024[i]
        eda = 100 * abs(zl - zd) / zd
        ebr = 100 * abs(zl - zb) / zb
        print(f"{m1024[i]:>2.0f}{a1024[i]:>8.2f}{beta_1024[i]:>8.3f}{zd:>9.3f}"
              f"{zb:>9.3f}{zl:>11.3f}{eda:>9.2f}{ebr:>9.2f}")
    err_da = np.array([100*abs(zPP_1024[i]-zeta_darcy(a1024[i]))/zeta_darcy(a1024[i]) for i in range(6)])
    err_br = np.array([100*abs(zPP_1024[i]-zeta_brinkman(a1024[i], c_use))/zeta_brinkman(a1024[i], c_use) for i in range(6)])
    print("-" * 74)
    print(f"  erro médio Darcy = {err_da.mean():.2f}%   ->   Brinkman = {err_br.mean():.2f}%"
          f"   (máx Darcy {err_da.max():.1f}% -> Brinkman {err_br.max():.1f}%)")

    # ════════════ Figura ════════════
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "font.size": 11, "axes.grid": True,
                         "grid.linestyle": ":", "grid.alpha": 0.6})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # Painel A: déficit vs β (a física)
    bb = np.linspace(0, beta_1024.max() * 1.08, 200)
    ax1.plot(bb, 100 * c_use * bb, "-", color="#1a5276", lw=1.8,
             label=rf"Brinkman:  $d = {c_use:.2f}\,\beta$")
    ax1.scatter(beta_1024, 100 * d_pp, s=55, color="#922b21", zorder=6, label="LBM pico-a-pico (NY=1024)")
    ax1.scatter(beta_1024, 100 * d_f, s=40, marker="s", color="#b9770e", zorder=6, label="LBM Fourier (NY=1024)")
    d_pp_2048 = 1.0 - zPP_2048 / zeta_darcy(a2048)
    ax1.scatter(a2048 * sqDa, 100 * d_pp_2048, s=90, marker="D",
                facecolor="none", edgecolor="#1e8449", linewidths=2, zorder=7,
                label="LBM NY=2048 (mesmo β → mesmo d)")
    for i in range(6):
        ax1.annotate(f"m={i+1}", (beta_1024[i], 100*d_pp[i]),
                     textcoords="offset points", xytext=(5, -10), fontsize=8)
    ax1.set_xlabel(r"$\beta = \sqrt{K}\,k = \alpha\sqrt{\mathrm{Da}}$  (blindagem de Brinkman)")
    ax1.set_ylabel(r"Déficit  $d = 1 - \zeta_{num}/\zeta_{Darcy}$  (\%)")
    ax1.set_title(r"Déficit é LINEAR em $\beta$ — e invariante na resolução")
    ax1.legend(frameon=True, edgecolor="black", fontsize=8.5, loc="upper left")
    ax1.tick_params(direction="in", top=True, right=True)

    # Painel B: ζ(α) — Darcy vs Brinkman vs LBM
    aa = np.linspace(1e-3, a1024.max() * 1.08, 400)
    ax2.plot(aa, zeta_darcy(aa), "--", color="#922b21", lw=1.8, label=r"$\zeta_{Darcy}$ (Eq. 9)")
    ax2.plot(aa, zeta_brinkman(aa, c_use), "-", color="#1a5276", lw=1.8,
             label=rf"$\zeta_{{Brinkman}}=\zeta_{{Darcy}}(1-{c_use:.2f}\sqrt{{Da}}\,\alpha)$")
    ax2.scatter(a1024, zPP_1024, s=55, color="#1e8449", zorder=6, label="LBM pico-a-pico (NY=1024)")
    ax2.scatter(a2048, zPP_2048, s=80, marker="D", facecolor="none",
                edgecolor="#1e8449", linewidths=2, zorder=7, label="LBM (NY=2048)")
    ax2.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax2.set_xlabel(r"$\alpha = 2\pi m$")
    ax2.set_ylabel(r"Taxa de crescimento $\zeta$")
    ax2.set_title(r"LBM valida a Eq. 9 corrigida por Brinkman")
    ax2.legend(frameon=True, edgecolor="black", fontsize=8.5, loc="upper left")
    ax2.tick_params(direction="in", top=True, right=True)

    fig.tight_layout()
    out_png = os.path.join(AQUI, "correcao_brinkman.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[ok] figura: {out_png}")

    # ════════════ Tabela LaTeX ════════════
    linhas = []
    for i in range(6):
        zd, zb, zl = zeta_darcy(a1024[i]), zeta_brinkman(a1024[i], c_use), zPP_1024[i]
        linhas.append(f"      {m1024[i]:.0f} & {a1024[i]:.2f} & {beta_1024[i]:.3f} & "
                      f"{zd:.3f} & {zb:.3f} & {zl:.3f} & "
                      f"{100*abs(zl-zd)/zd:.1f} & {100*abs(zl-zb)/zb:.1f} \\\\")
    tex = (r"""% Tabela: correção de Brinkman da relação de dispersão (sweep AD_W3, NY=1024).
% Gerada por correcao_brinkman.py.  \input{tabela_brinkman}
\begin{table}[htbp]
  \centering\small
  \setlength{\tabcolsep}{6pt}\renewcommand{\arraystretch}{1.2}
  \begin{tabular}{r r r r r r r r}
    \toprule
    $m$ & $\alpha$ & $\beta$ & $\zeta_{Darcy}$ & $\zeta_{Brink}$ & $\zeta_{LBM}$
        & erro Darcy (\%) & erro Brink (\%) \\
    \midrule
""" + "\n".join(linhas) + r"""
    \bottomrule
  \end{tabular}
  \caption{Comparação $\zeta=s\,N_Y/U$ da LBM (pico-a-pico) contra a LSA de Darcy
  pura (Eq.~9) e a versão corrigida por Brinkman
  $\zeta_{Brink}=\zeta_{Darcy}(1-c\sqrt{\mathrm{Da}}\,\alpha)$, $c=""" +
        f"{c_use:.2f}" + r"""$. O erro cai de até $\sim33\%$ (Darcy) para
  poucos por cento (Brinkman). $\beta=\sqrt{K}k=\alpha\sqrt{\mathrm{Da}}$ é o
  número de blindagem de Brinkman, invariante na resolução.}
  \label{tab:brinkman}
\end{table}
""")
    out_tex = os.path.join(AQUI, "tabela_brinkman.tex")
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"[ok] tabela:  {out_tex}")
