# -*- coding: utf-8 -*-
"""
gerar_fig_erro_vs_m.py — figura "erro cresce mais suave com m ao reduzir Da".
Usa a forma de correção Brinkman-√ calibrada nos 6 pontos NY=1024.
Gera: fig_erro_vs_m.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = os.path.dirname(os.path.abspath(__file__))

# Forma Brinkman-√ ajustada (déficit d = c·(√(β²+β₀²) − β₀), β = α√Da)
c, b0 = 0.9350, 0.0511
def d_pred(beta):
    return c * (np.sqrt(beta**2 + b0**2) - b0)

# Déficit medido (sweep NY=1024, pico-a-pico, Da=1.25e-4 ⇔ K₀=524 no NY=2048)
m_med = np.arange(1, 7, dtype=float)
d_med = np.array([0.00, 8.70, 17.69, 23.52, 28.80, 33.06])  # %

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                     "font.size": 11, "axes.grid": True,
                     "grid.linestyle": ":", "grid.alpha": 0.6})
fig, ax = plt.subplots(figsize=(7.4, 5.2))

mm = np.linspace(1, 4, 200)
cores = {524: "#922b21", 131: "#b9770e", 33: "#1e8449"}
rotulos = {524: r"$K_0{=}524$  (Da$=1.25\times10^{-4}$) — atual",
           131: r"$K_0{=}131$  (Da$=3.1\times10^{-5}$) — opção 1",
           33:  r"$K_0{=}33$   (Da$=7.9\times10^{-6}$) — opção 2"}
for K0 in (524, 131, 33):
    Da = K0 / 2048.0**2
    beta = 2 * np.pi * mm * np.sqrt(Da)
    ax.plot(mm, 100 * d_pred(beta), "-", color=cores[K0], lw=2.0, label=rotulos[K0])

# Pontos medidos (apenas K0=524 ⇔ Da atual)
ax.scatter(m_med[:4], d_med[:4], s=60, color=cores[524], zorder=6,
           edgecolor="black", linewidths=0.6, label="LBM medido (Da atual)")

ax.axhline(3.0, color="black", ls="--", lw=1.0, alpha=0.7)
ax.text(1.05, 3.4, "meta 3\\%", fontsize=9)
ax.set_xlabel(r"Modo $m$")
ax.set_ylabel(r"Erro relativo da taxa  $|d| = |1 - \zeta_{num}/\zeta_{Darcy}|$  (\%)")
ax.set_title(r"Reduzir Da achata a curva erro-vs-$m$  ($\beta=\alpha\sqrt{\mathrm{Da}}$)")
ax.set_xticks([1, 2, 3, 4])
ax.set_xlim(1, 4)
ax.set_ylim(0, 25)
ax.legend(frameon=True, edgecolor="black", fontsize=9, loc="upper left")
ax.tick_params(direction="in", top=True, right=True)

fig.tight_layout()
out = os.path.join(AQUI, "fig_erro_vs_m.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print("[ok]", out)
