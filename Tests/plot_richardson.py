import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_convergence():
    df = pd.read_csv('convergence_results.csv').dropna()
    df = df.sort_values('nx')

    dx = 1.0 / df['nx'].values
    phi_metric = df['tip_position_normalized'].values

    # Configuração de estilo acadêmico
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- SUBPLOT 1: Assíntota de Malha ---
    axes[0].plot(df['nx'], phi_metric, 'ko-', linewidth=1.5, markersize=6)
    axes[0].set_xlabel(r'Resolução da Malha ($N_x$)')
    axes[0].set_ylabel(r'Posição Adimensional da Ponta ($x_{tip}/L_x$)')
    axes[0].set_title('Independência de Malha')

    # Destacar a convergência para um valor assintótico
    axes[0].axhline(y=phi_metric[-1], color='r', linestyle='--', alpha=0.5, label='Valor Assintótico')
    axes[0].legend()

    # --- SUBPLOT 2: Extrapolação de Richardson (Taxa de Erro) ---
    # Erro relativo entre resoluções sucessivas
    erros = np.abs(np.diff(phi_metric))
    d_inv = dx[:-1]  # Tamanho característico da malha (\Delta x)

    axes[1].loglog(d_inv, erros, 'bo-', linewidth=2, markersize=8, label='Erro Simulado')

    # Adicionando a linha de referência teórica de 2ª Ordem: O(dx^2)
    # Ajusta a altura da linha de referência para ficar próxima aos dados reais
    if len(erros) > 1:
        C = erros[0] / (d_inv[0] ** 2)
        ref_order2 = C * (d_inv ** 2)
        axes[1].loglog(d_inv, ref_order2, 'k--', linewidth=1.5, label=r'Teórico $\mathcal{O}(\Delta x^2)$')

    axes[1].set_xlabel(r'Espaçamento da Malha ($\Delta x$)')
    axes[1].set_ylabel(r'Erro Relativo de Truncamento $|f_{k} - f_{k-1}|$')
    axes[1].set_title('Análise de Erro (Log-Log)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('analise_convergencia_dissertacao.png', dpi=300, bbox_inches='tight')
    print("Gráfico acadêmico salvo em: analise_convergencia_dissertacao.png")


if __name__ == "__main__":
    plot_convergence()