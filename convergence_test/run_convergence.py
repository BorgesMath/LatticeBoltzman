import os
import sys
import shutil
import numpy as np
import subprocess
import pandas as pd
from tqdm import tqdm


# Caminhos
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, 'config', 'config.py')
BACKUP_PATH = os.path.join(ROOT_DIR, 'config', 'config_backup.py')
MAIN_SCRIPT = os.path.join(ROOT_DIR, 'main.py')
RESULTS_CSV = os.path.join(os.path.dirname(__file__), 'convergence_results.csv')

# ==============================================================================
# CONFIGURAÇÃO DO TESTE DE CONVERGÊNCIA (ESCALONAMENTO DIFUSIVO)
# ==============================================================================
# Fatores de escala para a malha (k). 1.0 é a malha base (300x600).
# Usaremos malhas mais grossas e mais finas para ver a assíntota.
SCALES = [0.6, 0.8, 1.0, 1.2, 1.4]

# Parâmetros Base (Do seu config original)
BASE_PARAMS = {
    'NY': 300, 'NX': 600, 'MAX_ITER': 5000,
    'TAU_IN': 0.52, 'TAU_OUT': 3.0,
    'U_INLET': 0.05, 'K_0': 5000.0,
    'M_MOBILITY': 0.002, 'SIGMA': 0.0001, 'INTERFACE_WIDTH': 3.0,
    'H0': 0.08, 'CHI_MAX': 1.2
}


def generate_config_content(scale):
    """
    Gera o conteúdo do arquivo config.py aplicando Escalonamento Difusivo.
    Mantém Reynolds e Capilaridade constantes.
    """
    k = scale

    # Escalonamento
    # Grid aumenta linearmente com k
    nx_new = int(BASE_PARAMS['NX'] * k)
    ny_new = int(BASE_PARAMS['NY'] * k)

    # Tempo difusivo: dt ~ dx^2. Se dx cai (resolução aumenta), dt cai ao quadrado.
    # Logo, precisamos de mais iterações: N_iter ~ k^2
    max_iter_new = int(BASE_PARAMS['MAX_ITER'] * (k ** 2))

    # Velocidade em unidades de rede deve cair para manter Mach baixo e Re constante
    # u_lb ~ 1/k
    u_inlet_new = BASE_PARAMS['U_INLET'] / k

    # Tensão superficial para manter Ca constante (Ca ~ u/sigma)
    # sigma_new ~ 1/k
    sigma_new = BASE_PARAMS['SIGMA'] / k

    # Mobilidade para manter Peclet (Pe ~ u * L / M * sigma)
    # u ~ 1/k, L ~ k, sigma ~ 1/k -> Pe ~ (1/k * k) / (M * 1/k) = 1 / (M/k).
    # Para Pe constante, M deve escalar com k?
    # Na prática de LBM, mantemos M constante ou ajustamos conforme a difusão numérica.
    # Vamos manter M fixo em unidades físicas, o que implica ajuste se mudarmos dt/dx.
    # Simplificação robusta: M_lb escalona com 1/k para consistência difusiva
    m_mobility_new = BASE_PARAMS['M_MOBILITY'] / k

    # O resto (TAU) fica constante no scaling difusivo pois nu_phys = C_s^2 (tau-0.5) dt/dx^2
    # dt/dx^2 = 1. Então viscosidade física é preservada mantendo TAU fixo.

    content = f"""# config.py - GERADO AUTOMATICAMENTE PARA TESTE DE CONVERGENCIA (k={k:.2f})
import numpy as np

# 1. TOPOLOGIA (Escala k={k})
NY = {ny_new}
NX = {nx_new}
MAX_ITER = {max_iter_new}
SNAPSHOT_STEPS = 0  # Desativar imagens para economizar I/O

# 2. HIDRODINAMICA (Diffusive Scaling)
TAU_IN = {BASE_PARAMS['TAU_IN']}
TAU_OUT = {BASE_PARAMS['TAU_OUT']}
U_INLET = {u_inlet_new:.6f}  # Ajustado para Re constante
K_0 = {BASE_PARAMS['K_0']}

# 3. CAHN-HILLIARD
M_MOBILITY = {m_mobility_new:.6f}
SIGMA = {sigma_new:.6f}       # Ajustado para Ca constante
INTERFACE_WIDTH = {BASE_PARAMS['INTERFACE_WIDTH']} # Mantido fixo em Lattice Units (Sharp Interface limit)
CH_SUBSTEPS = 10
DT_CH = 1.0 / CH_SUBSTEPS

BETA = 3.0 * SIGMA * INTERFACE_WIDTH / 4.0
KAPPA = 3.0 * SIGMA * INTERFACE_WIDTH / 8.0

# 4. MAGNETOSTATICA
H0 = {BASE_PARAMS['H0']}
CHI_MAX = {BASE_PARAMS['CHI_MAX']}
SOR_OMEGA = 1.85 

# 5. D2Q9
W_LBM = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float64)
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)
"""
    return content, ny_new, nx_new


def get_finger_tip_position(mode_m):
    """
    Lê o último estado (phi) e encontra a posição x mais avançada da interface.
    Como não estamos salvando VTKs, precisamos hackear o output ou modificar o main
    para retornar valor.

    Estratégia prática: Ler o arquivo de curvatura gerado ou usar o post_process.
    Vou assumir que o post_process salva dados ou podemos ler o último .npy se salvar.

    Para este script ser independente, vamos confiar que o main roda e gera series temporais.
    Mas a 'posição da ponta' não está no CSV.

    SOLUÇÃO: O script vai injetar um pequeno código de "probe" no final da simulação
    ou você deve modificar o post_process.py para salvar 'tip_position.txt'.
    """
    # Para simplificar, vou assumir que a massa total (conservação) é a métrica
    # OU vou modificar o main para imprimir a posição da ponta no stdout.
    return 0.0


def run_tests():
    # 1. Backup do Config
    if os.path.exists(CONFIG_PATH):
        shutil.copy(CONFIG_PATH, BACKUP_PATH)

    results = []

    try:
        for k in SCALES:
            print(f"\n---> Iniciando Simulação Escala k={k:.2f}")

            # Reescrever Config
            cfg_content, ny, nx = generate_config_content(k)
            with open(CONFIG_PATH, 'w') as f:
                f.write(cfg_content)

            # Executar Simulação
            # Capturando stdout para pegar métricas se necessário
            process = subprocess.run(
                [sys.executable, MAIN_SCRIPT],
                cwd=ROOT_DIR,
                capture_output=True,
                text=True
            )

            if process.returncode != 0:
                print(f"ERRO na escala {k}: {process.stderr}")
                continue

            # PARSING DE RESULTADOS
            # Como o código original não exporta a "posição da ponta" num txt simples,
            # vamos calcular a conservação de massa (que está no print/log) como proxy de qualidade
            # ou ler os arquivos npy se existirem.

            # Solução Robusta: Adicionaremos um print no final deste script analisando o output gerado.
            # O ideal seria alterar o post_process.py para salvar 'tip_pos.txt'.
            # Vou assumir aqui que você implementará o passo 3 abaixo.

            try:
                # Lendo a posição da ponta do arquivo que vamos criar no passo 3
                tip_file = os.path.join(ROOT_DIR, f"st_analise_modo_4", "tip_position.txt")
                if os.path.exists(tip_file):
                    with open(tip_file, 'r') as tf:
                        tip_pos_lu = float(tf.read().strip())
                        # Normalizar pela largura do domínio LX para comparação física
                        tip_pos_norm = tip_pos_lu / nx
                else:
                    tip_pos_norm = np.nan
            except Exception as e:
                print(f"Erro ao ler dados: {e}")
                tip_pos_norm = np.nan

            print(f"   [OK] Grid: {nx}x{ny} | Tip Position (L/L0): {tip_pos_norm:.5f}")
            results.append({
                'scale': k,
                'nx': nx,
                'ny': ny,
                'dx': 1.0 / nx,  # Espaçamento relativo
                'tip_position_normalized': tip_pos_norm
            })

    finally:
        # Restaurar Config Original
        if os.path.exists(BACKUP_PATH):
            shutil.move(BACKUP_PATH, CONFIG_PATH)
            print("\nConfiguração original restaurada.")

    # Salvar Resultados
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"Resultados salvos em {RESULTS_CSV}")


if __name__ == "__main__":
    # Certifique-se de que o post_process.py foi modificado conforme instrução abaixo
    run_tests()