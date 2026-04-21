import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_convergence():
    df = pd.read_csv('convergence_results.csv')
    df = df.sort_values('nx')

    # Variáveis
    dx = 1.0 / df['nx'].values
    phi_metric = df['tip_position_normalized'].values

    # Cálculo da Ordem de Convergência (p)
    # Assumindo 3 malhas mais finas: f1 (fina), f2 (média), f3 (grossa)
    # r = razão de refino (aprox constante se usarmos escalas 1.0, 1.2, 1.4)
    # p = ln((f3 - f2) / (f2 - f1)) / ln(r)

    plt.figure(figsize=(10, 6))

    # 1. Gráfico da Métrica vs 1/NX
    plt.subplot(1, 2, 1)
    plt.plot(df['nx'], phi_metric, 'o-', linewidth=2, color='navy')
    plt.xlabel('Resolução (Nx)')
    plt.ylabel('Posição da Ponta Normalizada (x/L)')
    plt.title('Convergência de Malha')
    plt.grid(True, alpha=0.3)

    # 2. Extrapolação de Richardson (Assintótica)
    # Plotando Erro Relativo entre passos
    erros = np.abs(np.diff(phi_metric))
    d_inv = 1. / df['nx'].values[:-1]  # 1/Nx

    plt.subplot(1, 2, 2)
    plt.loglog(d_inv, erros, 'r-o')
    plt.xlabel(r'Tamanho da Malha ($1/N_x$)')
    plt.ylabel('Diferença Relativa $|f_{fine} - f_{coarse}|$')
    plt.title('Taxa de Convergência')
    plt.grid(True, which="both", ls="-", alpha=0.3)

    plt.tight_layout()
    plt.savefig('analise_convergencia.png', dpi=300)
    print("Gráfico salvo em analise_convergencia.png")


if __name__ == "__main__":
    plot_convergence()