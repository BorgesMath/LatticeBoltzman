# post_process/analise_caso.py
"""
Análise não-linear de um caso de digitação viscosa/magnética (regime
plenamente desenvolvido, FORA do limite linear da LSA).

Lê TODOS os .vtr de <case_dir>/vtk uma única vez e extrai, por instante:

    • A_pp        — semi-amplitude pico-a-pico da interface φ=0      [l.u.]
    • A_fourier   — projeção de Fourier do modo m (compatível LSA)   [l.u.]
    • x_tip       — penetração máxima do dedo  (max_y x*(y))         [l.u.]
    • x_raiz      — recuo máximo               (min_y x*(y))         [l.u.]
    • x_med       — posição média da frente                         [l.u.]
    • u_max,u_med — velocidade de rede máxima/média                 [l.u.]
    • Ma = u_max/cs  — número de Mach local (cs = 1/√3)             [-]
    • dp          — queda de pressão inlet-outlet (p = ρ/3)         [l.u.]
    • massa       — Σρ  (diagnóstico de conservação)               [l.u.]
    • S_in        — fração de área do fluido injetado (φ>0)         [-]
    • L_int       — comprimento da interface (arestas de cruzamento) [l.u.]

Gera (em --outdir, NUNCA em post_process/):
    fig_morfologia.png   fig_integridade.png   fig_amplitude.png
    fig_transporte.png   fig_magnetico.png     resumo_caso.json
e cacheia em <case_dir>:
    curvatura_temporal.npz  (t, amplitude  ->  reutilizado por valida_lsa.py)
    diagnostico_caso.npz    (todas as séries acima)

Uso:
    python post_process/analise_caso.py <case_dir> --outdir <pasta_figuras>
"""

import os
import re
import sys
import glob
import json
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import vtk
from vtk.util import numpy_support as ns

CS = 1.0 / np.sqrt(3.0)   # velocidade do som no reticulado D2Q9
CS2 = 1.0 / 3.0           # p = cs² · ρ

# ── estilo acadêmico (igual a valida_lsa.py) ──────────────────────────
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12.5,
    "legend.fontsize": 9, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "lines.linewidth": 1.6, "axes.grid": True,
    "grid.linestyle": ":", "grid.alpha": 0.55,
})


# ═══════════════════════════════════════════════════════════════════
#  Leitura de VTR
# ═══════════════════════════════════════════════════════════════════
def _ts(fname):
    m = re.search(r"dados_macro_(\d+)\.vtr$", os.path.basename(fname))
    return int(m.group(1)) if m else -1


def _ler_vtr(path):
    """Devolve dict de campos (ny,nx) ou (ny,nx,3); None se ilegível."""
    try:
        r = vtk.vtkXMLRectilinearGridReader()
        r.SetFileName(path)
        r.Update()
        g = r.GetOutput()
        dims = g.GetDimensions()
        nx, ny = dims[0] - 1, dims[1] - 1
        if nx <= 0 or ny <= 0:
            return None
        cd = g.GetCellData()
        out = {"nx": nx, "ny": ny}
        for nome in ("fase_phi", "densidade_rho",
                     "potencial_perturbacao_psi_tilde", "potencial_total_psi"):
            a = cd.GetArray(nome)
            if a is not None:
                out[nome] = ns.vtk_to_numpy(a).reshape((ny, nx))
        for nome in ("velocidade", "campo_magnetico_H"):
            a = cd.GetArray(nome)
            if a is not None:
                out[nome] = ns.vtk_to_numpy(a).reshape((ny, nx, 3))
        if "fase_phi" not in out:
            return None
        return out
    except Exception as e:
        print(f"   [skip] {os.path.basename(path)} ilegível: {e}")
        return None


def _interface_x(phi):
    """x*(y) sub-pixel do 1º cruzamento φ=0 (esq.→dir.); array por linha válida."""
    ny, nx = phi.shape
    idx_r = np.argmax(phi < 0.0, axis=1)
    valid = idx_r > 0
    if not np.any(valid):
        return None
    ir = idx_r[valid]
    il = ir - 1
    linhas = np.arange(ny)[valid]
    phir = phi[linhas, ir]
    phil = phi[linhas, il]
    x = il + phil / (phil - phir + 1e-15)
    return linhas, x


def _amp_fourier(phi, m):
    res = _interface_x(phi)
    if res is None:
        return 0.0
    linhas, x = res
    ny = phi.shape[0]
    fase = np.exp(-2j * np.pi * m * linhas / ny)
    return float((2.0 / ny) * abs(np.sum(x * fase)))


def _comprimento_interface(phi):
    """Nº de arestas de célula onde φ troca de sinal ≈ perímetro do contorno φ=0."""
    sx = np.sum((phi[:, 1:] * phi[:, :-1]) < 0.0)
    sy = np.sum((phi[1:, :] * phi[:-1, :]) < 0.0)
    return float(sx + sy)


# ═══════════════════════════════════════════════════════════════════
#  Varredura temporal
# ═══════════════════════════════════════════════════════════════════
def varrer(case_dir, mode_m, n_morf=6):
    vtrs = sorted(glob.glob(os.path.join(case_dir, "vtk", "dados_macro_*.vtr")),
                  key=_ts)
    if not vtrs:
        raise FileNotFoundError(f"Nenhum .vtr em {case_dir}/vtk")
    n = len(vtrs)
    idx_morf = set(np.linspace(0, n - 1, n_morf, dtype=int).tolist())

    series = {k: [] for k in
              ("t", "A_pp", "A_fourier", "x_tip", "x_raiz", "x_med", "x_frente_vol",
               "u_max", "u_med", "Ma", "dp", "massa", "S_in", "L_int")}
    morf = []          # (t, phi)  para a montagem morfológica
    mapa_final = None  # último snapshot legível p/ figura magnética

    print(f"Lendo {n} VTRs de {case_dir} ...")
    for i, fp in enumerate(vtrs):
        d = _ler_vtr(fp)
        if d is None:
            continue
        t = _ts(fp)
        phi = d["fase_phi"]
        nx = d["nx"]

        res = _interface_x(phi)
        if res is not None:
            _, x = res
            x_tip, x_raiz, x_med = float(x.max()), float(x.min()), float(x.mean())
            A_pp = 0.5 * (x_tip - x_raiz)
        else:
            x_tip = x_raiz = x_med = A_pp = np.nan

        rho = d.get("densidade_rho")
        if rho is not None:
            massa = float(rho.sum())
            p_in = float(rho[:, 3:8].mean()) * CS2
            p_out = float(rho[:, nx - 8:nx - 3].mean()) * CS2
            dp = p_in - p_out
        else:
            massa = dp = np.nan

        vel = d.get("velocidade")
        if vel is not None:
            spd = np.sqrt(vel[:, :, 0] ** 2 + vel[:, :, 1] ** 2 + vel[:, :, 2] ** 2)
            u_max, u_med = float(spd.max()), float(spd.mean())
        else:
            u_max = u_med = np.nan

        series["t"].append(t)
        series["A_pp"].append(A_pp)
        series["A_fourier"].append(_amp_fourier(phi, mode_m))
        series["x_tip"].append(x_tip)
        series["x_raiz"].append(x_raiz)
        series["x_med"].append(x_med)
        series["u_max"].append(u_max)
        series["u_med"].append(u_med)
        series["Ma"].append(u_max / CS if np.isfinite(u_max) else np.nan)
        series["dp"].append(dp)
        series["massa"].append(massa)
        s_in = float(np.mean(phi > 0.0))
        series["S_in"].append(s_in)
        # Frente média ROBUSTA por volume = (área de fluido injetado)/NY = S_in·NX.
        # Monotônica com a injeção (imune a filamentos presos que corrompem x_med).
        series["x_frente_vol"].append(s_in * nx)
        series["L_int"].append(_comprimento_interface(phi))

        if i in idx_morf:
            morf.append((t, phi.copy()))
        mapa_final = (t, d)
        print(f"  [{i+1:3d}/{n}] t={t:6d}  A_pp={A_pp:7.2f}  x_tip={x_tip:7.1f}"
              f"  Ma={series['Ma'][-1]:.4f}  dp={dp:+.3e}  S_in={series['S_in'][-1]:.3f}")

    for k in series:
        series[k] = np.array(series[k], dtype=float)
    return series, morf, mapa_final, n


# ═══════════════════════════════════════════════════════════════════
#  Figuras
# ═══════════════════════════════════════════════════════════════════
def fig_morfologia(morf, outdir):
    """Montagem da evolução de φ (recortada na região da interface)."""
    # janela x comum a todos os instantes mostrados
    x_lo, x_hi = np.inf, -np.inf
    for _, phi in morf:
        res = _interface_x(phi)
        if res is not None:
            _, x = res
            x_lo = min(x_lo, x.min())
            x_hi = max(x_hi, x.max())
    ny = morf[0][1].shape[0]
    x0 = max(0, int(x_lo) - 40)
    x1 = min(morf[0][1].shape[1], int(x_hi) + 40)

    k = len(morf)
    fig, axes = plt.subplots(k, 1, figsize=(9.0, 1.7 * k + 0.6), squeeze=False)
    for ax, (t, phi) in zip(axes[:, 0], morf):
        sub = phi[:, x0:x1]
        im = ax.imshow(sub, origin="lower", aspect="auto",
                       extent=[x0, x1, 0, ny], cmap="coolwarm",
                       vmin=-1, vmax=1, interpolation="nearest")
        ax.contour(np.linspace(x0, x1, sub.shape[1]),
                   np.linspace(0, ny, sub.shape[0]), sub,
                   levels=[0.0], colors="k", linewidths=0.8)
        ax.set_ylabel(f"$t={t}$\n$y$ (l.u.)", fontsize=8.5)
        ax.tick_params(direction="in")
    axes[-1, 0].set_xlabel(r"$x$ (l.u.)  —  fluido injetado (vermelho, $\phi>0$) desloca o resistente (azul, $\phi<0$)")
    cbar = fig.colorbar(im, ax=axes[:, 0], fraction=0.012, pad=0.01)
    cbar.set_label(r"$\phi$")
    fig.suptitle("Evolução morfológica da interface — digitação magnética (campo normal)",
                 fontsize=12, y=0.995)
    out = os.path.join(outdir, "fig_morfologia.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def fig_integridade(s, params, outdir):
    """massa(t) e Ma(t) — validação do controle de Mach."""
    t = s["t"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))

    m0 = s["massa"][0]
    drift = (s["massa"] - m0) / m0 * 100.0
    ax1.plot(t, drift, "-o", ms=3, color="#1a5276")
    ax1.axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax1.set_xlabel(r"Passo de tempo $t$")
    ax1.set_ylabel(r"Deriva de massa $\Delta m/m_0$ (%)")
    ax1.set_title(f"Conservação de massa (final: {drift[-1]:+.3f}%)")
    ax1.tick_params(direction="in", top=True, right=True)

    ax2.plot(t, s["Ma"], "-o", ms=3, color="#922b21", label=r"$\mathrm{Ma}=u_{\max}/c_s$")
    ax2.axhline(0.1, color="#888", ls="--", lw=1.0, label=r"limite LBM $\mathrm{Ma}=0.1$")
    Ma_bulk = params["u_in"] / CS
    ax2.axhline(Ma_bulk, color="#1e8449", ls=":", lw=1.2,
                label=fr"$\mathrm{{Ma}}_{{bulk}}=U/c_s={Ma_bulk:.4f}$")
    ax2.set_xlabel(r"Passo de tempo $t$")
    ax2.set_ylabel(r"Número de Mach local")
    ax2.set_title(f"Mach (pico: {np.nanmax(s['Ma']):.3f})")
    ax2.legend(loc="best", frameon=True, edgecolor="black")
    ax2.tick_params(direction="in", top=True, right=True)

    fig.tight_layout()
    out = os.path.join(outdir, "fig_integridade.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def fig_amplitude(s, params, s_ana, outdir):
    """A_pp e A_fourier x envelope LSA analítico (escala log)."""
    t = s["t"]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.semilogy(t, s["A_pp"], "-o", ms=3.5, color="#1a5276",
                label=r"$A_{pp}$ pico-a-pico (excursão do dedo)")
    ax.semilogy(t, s["A_fourier"], "-s", ms=3.5, color="#1e8449",
                label=fr"$A_{{m}}$ Fourier modo $m={params['mode_m']}$")
    # envelope analítico ancorado no 1º ponto
    pos = s["A_pp"] > 0
    if np.any(pos):
        A0 = s["A_pp"][pos][0]
        t0 = t[pos][0]
        env = A0 * np.exp(s_ana * (t - t0))
        ax.semilogy(t, env, "--", color="#922b21", lw=1.6,
                    label=fr"Envelope LSA $\propto e^{{s_{{ana}}t}}$, $s_{{ana}}={s_ana:.2e}$")
    ax.set_ylim(1.0, max(np.nanmax(s["A_pp"]) * 3, 1e3))
    ax.set_xlabel(r"Passo de tempo $t$")
    ax.set_ylabel(r"Amplitude (l.u.)")
    ax.set_title("Crescimento da perturbação: LSA prevê blow-up quase imediato")
    ax.legend(loc="lower right", frameon=True, edgecolor="black")
    ax.tick_params(direction="in", top=True, right=True)
    fig.tight_layout()
    out = os.path.join(outdir, "fig_amplitude.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def fig_transporte(s, params, outdir):
    """Δp(t), avanço da frente e saturação — relação vazão(fixa)×pressão."""
    t = s["t"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))

    ax1.plot(t, s["dp"], "-o", ms=3, color="#6c3483")
    ax1.axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax1.set_xlabel(r"Passo de tempo $t$")
    ax1.set_ylabel(r"$\Delta p = p_{in}-p_{out}$ (l.u.)")
    ax1.set_title("Queda de pressão (vazão imposta no inlet)")
    ax1.tick_params(direction="in", top=True, right=True)

    ax2.plot(t, s["x_tip"], "-", color="#922b21", label=r"$x_{tip}$ (ponta do dedo)")
    ax2.plot(t, s["x_frente_vol"], "-", color="#1a5276", lw=2.2,
             label=r"$x_{frente}=S_{in}N_X$ (volume, robusta)")
    ax2.plot(t, s["x_med"], "--", color="#5499c7", lw=1.2,
             label=r"$x_{méd}$ (1ª travessia $\phi<0$)")
    ax2.plot(t, s["x_raiz"], ":", color="#1e8449", lw=1.2, label=r"$x_{raiz}$ (cavidade)")
    ax2.set_xlabel(r"Passo de tempo $t$")
    ax2.set_ylabel(r"Posição da interface $x$ (l.u.)")
    ax2.set_title("Avanço da frente")
    ax2.legend(loc="best", frameon=True, edgecolor="black")
    ax2.tick_params(direction="in", top=True, right=True)
    ax2b = ax2.twinx()
    ax2b.plot(t, s["S_in"], ":", color="#888", lw=1.4)
    ax2b.set_ylabel(r"Saturação $S_{in}$ ($\phi>0$)", color="#666")
    ax2b.tick_params(axis="y", colors="#666")
    ax2b.grid(False)

    fig.tight_layout()
    out = os.path.join(outdir, "fig_transporte.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def fig_magnetico(mapa_final, outdir):
    """|H| e ψ̃ no último instante, com a interface sobreposta."""
    t, d = mapa_final
    phi = d["fase_phi"]
    ny, nx = phi.shape
    H = d.get("campo_magnetico_H")
    psi = d.get("potencial_perturbacao_psi_tilde")
    if H is None and psi is None:
        print("[info] sem campos magnéticos no VTR — fig_magnetico pulada.")
        return

    # recorte na região da interface
    res = _interface_x(phi)
    if res is not None:
        _, x = res
        x0 = max(0, int(x.min()) - 60)
        x1 = min(nx, int(x.max()) + 60)
    else:
        x0, x1 = 0, nx
    xs = np.linspace(x0, x1, x1 - x0)
    ys = np.linspace(0, ny, ny)
    sub_phi = phi[:, x0:x1]

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.4))

    if H is not None:
        magH = np.sqrt(H[:, :, 0] ** 2 + H[:, :, 1] ** 2)[:, x0:x1]
        im0 = axes[0].imshow(magH, origin="lower", aspect="auto",
                             extent=[x0, x1, 0, ny], cmap="magma")
        axes[0].contour(xs, ys, sub_phi, levels=[0.0], colors="cyan", linewidths=0.9)
        fig.colorbar(im0, ax=axes[0], fraction=0.012, pad=0.01).set_label(r"$|\mathbf{H}|$")
        axes[0].set_title(r"Intensidade do campo $|\mathbf{H}|$ (concentração nos dedos)")
        axes[0].set_ylabel(r"$y$ (l.u.)")
    if psi is not None:
        sub_psi = psi[:, x0:x1]
        vmax = np.percentile(np.abs(sub_psi), 99) + 1e-30
        im1 = axes[1].imshow(sub_psi, origin="lower", aspect="auto",
                             extent=[x0, x1, 0, ny], cmap="RdBu_r",
                             norm=TwoSlopeNorm(0.0, -vmax, vmax))
        axes[1].contour(xs, ys, sub_phi, levels=[0.0], colors="k", linewidths=0.9)
        fig.colorbar(im1, ax=axes[1], fraction=0.012, pad=0.01).set_label(r"$\tilde\psi$")
        axes[1].set_title(r"Potencial de perturbação magnética $\tilde\psi$")
        axes[1].set_ylabel(r"$y$ (l.u.)")
    axes[-1].set_xlabel(r"$x$ (l.u.)")
    fig.suptitle(f"Campo magnético no instante final $t={t}$", fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(outdir, "fig_magnetico.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Análise não-linear de um caso LBM.")
    ap.add_argument("case_dir")
    ap.add_argument("--outdir", required=True, help="Pasta de saída das figuras/JSON.")
    ap.add_argument("--n-morf", type=int, default=6)
    args = ap.parse_args()

    # importa a física da LSA já validada
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import valida_lsa as vl

    case_dir = args.case_dir
    os.makedirs(args.outdir, exist_ok=True)
    params = vl.extrair_parametros(case_dir)
    zeta_ana = vl.zeta_analitico(
        params["alpha_sim"], params["M"], params["Bo"], params["Ca_m"],
        params["Lambda_m"], params["H0n_sq"], params["H0t_sq"],
        params["Da"], params["Ca"])
    s_ana = zeta_ana * params["U_ref"] / params["L_ref"]

    s, morf, mapa_final, n = varrer(case_dir, params["mode_m"], args.n_morf)

    # caches no diretório do caso (padrão do projeto; NÃO em post_process)
    np.savez(os.path.join(case_dir, "curvatura_temporal.npz"),
             t=s["t"], amplitude=s["A_fourier"])
    np.savez(os.path.join(case_dir, "diagnostico_caso.npz"), **s)

    # figuras
    fig_morfologia(morf, args.outdir)
    fig_integridade(s, params, args.outdir)
    fig_amplitude(s, params, s_ana, args.outdir)
    fig_transporte(s, params, args.outdir)
    fig_magnetico(mapa_final, args.outdir)

    # resumo escalar p/ a tabela LaTeX
    resumo = dict(
        id_caso=os.path.basename(os.path.normpath(case_dir)),
        n_snapshots=int(n),
        mode_m=int(params["mode_m"]),
        M=params["M"], Da=params["Da"], Ca=params["Ca"],
        Bo_mag=params["Ca_m"], Lambda_m=params["Lambda_m"],
        zeta_ana=float(zeta_ana), s_ana=float(s_ana),
        massa_drift_pct=float((s["massa"][-1] - s["massa"][0]) / s["massa"][0] * 100),
        Ma_pico=float(np.nanmax(s["Ma"])),
        Ma_bulk=float(params["u_in"] / CS),
        A_pp_final=float(s["A_pp"][-1]),
        A_pp_frac_NY=float(s["A_pp"][-1] / params["ny"]),
        x_tip_final=float(s["x_tip"][-1]),
        S_in_final=float(s["S_in"][-1]),
        x_frente_vol_inicial=float(s["x_frente_vol"][0]),
        x_frente_vol_final=float(s["x_frente_vol"][-1]),
        L_int_final=float(s["L_int"][-1]),
        L_int_inicial=float(s["L_int"][0]),
        dp_inicial=float(s["dp"][0]),
        dp_final=float(s["dp"][-1]),
        dp_medio=float(np.nanmean(s["dp"])),
    )
    with open(os.path.join(args.outdir, "resumo_caso.json"), "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)
    print("\n=== RESUMO ===")
    for k, v in resumo.items():
        print(f"  {k:18s} = {v}")
    print(f"\n[OK] JSON: {os.path.join(args.outdir, 'resumo_caso.json')}")


if __name__ == "__main__":
    main()
