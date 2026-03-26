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
SCALES = [0.6, 0.8, 1.0, 1.2, 1.4]

# Parâmetros Base
BASE_PARAMS = {
    'NY': 300, 'NX': 600, 'MAX_ITER': 5000,
    'TAU_IN': 1.0, 'TAU_OUT': 3.0,
    'U_INLET': 0.1, 'K_0': 5000.0,
    'M_MOBILITY': 0.002, 'SIGMA': 0.0001, 'INTERFACE_WIDTH': 3.0,
    'H0': 0.08, 'CHI_MAX': 1.2,
    'METODO_MAGNETISMO': "'POISSON'",
    'H_ANGLE': 0.0
}


def generate_config_content(scale):
    k = scale

    # 1. Escalonamento Espacial e Temporal
    nx_new = int(BASE_PARAMS['NX'] * k)
    ny_new = int(BASE_PARAMS['NY'] * k)
    max_iter_new = int(BASE_PARAMS['MAX_ITER'] * (k ** 2))

    # 2. Escalonamento Cinemático e Dinâmico
    u_inlet_new = BASE_PARAMS['U_INLET'] / k
    sigma_new = BASE_PARAMS['SIGMA'] / k
    m_mobility_new = BASE_PARAMS['M_MOBILITY']

    # 3. Escalonamento da Interface (CORREÇÃO FUNDAMENTAL)
    # A espessura da interface em nós da malha DEVE escalar linearmente com k
    # para que a espessura física real permaneça invariante em todas as simulações.
    interface_width_new = BASE_PARAMS['INTERFACE_WIDTH'] * k

    content = f"""# config.py - GERADO AUTOMATICAMENTE PARA TESTE DE CONVERGENCIA (k={k:.2f})
import numpy as np

# 1. TOPOLOGIA (Escala k={k})
NY = {ny_new}
NX = {nx_new}
MAX_ITER = {max_iter_new}
SNAPSHOT_STEPS = 0  # Desativar imagens para economizar I/O

# 2. HIDRODINAMICA
TAU_IN = {BASE_PARAMS['TAU_IN']}
TAU_OUT = {BASE_PARAMS['TAU_OUT']}
U_INLET = {u_inlet_new:.6f}
K_0 = {BASE_PARAMS['K_0']}

# 3. CAHN-HILLIARD
M_MOBILITY = {m_mobility_new:.6f}
SIGMA = {sigma_new:.6f}
INTERFACE_WIDTH = {interface_width_new:.6f}
CH_SUBSTEPS = 10
DT_CH = 1.0 / CH_SUBSTEPS

# Nota Analítica: Como SIGMA decai com 1/k e INTERFACE_WIDTH cresce com k, 
# o produto permanece constante. Logo, BETA e KAPPA (física do Cahn-Hilliard) são preservados.
BETA = 3.0 * SIGMA * INTERFACE_WIDTH / 4.0
KAPPA = 3.0 * SIGMA * INTERFACE_WIDTH / 8.0

# 4. MAGNETOSTATICA E CONTROLE DE SOLVER
METODO_MAGNETISMO = {BASE_PARAMS['METODO_MAGNETISMO']}
H0 = {BASE_PARAMS['H0']}
CHI_MAX = {BASE_PARAMS['CHI_MAX']}
SOR_OMEGA = 1.85 
H_ANGLE = {BASE_PARAMS['H_ANGLE']}

# 5. D2Q9
W_LBM = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float64)
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)
"""
    return content, ny_new, nx_new


def run_tests():
    if os.path.exists(CONFIG_PATH):
        shutil.copy(CONFIG_PATH, BACKUP_PATH)

    results = []

    try:
        for k in SCALES:
            print(f"\n---> Iniciando Simulação Escala k={k:.2f}")

            cfg_content, ny, nx = generate_config_content(k)
            with open(CONFIG_PATH, 'w') as f:
                f.write(cfg_content)

            process = subprocess.run(
                [sys.executable, MAIN_SCRIPT],
                cwd=ROOT_DIR,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )

            if process.returncode != 0:
                print(f"ERRO na escala {k}:\n{process.stderr}")
                continue

            try:
                tip_file = os.path.join(ROOT_DIR, f"st_analise_modo_32", "tip_position.txt")
                if not os.path.exists(tip_file):
                    tip_file = os.path.join(ROOT_DIR, "st_analise_modo_4", "tip_position.txt")

                if os.path.exists(tip_file):
                    with open(tip_file, 'r') as tf:
                        tip_pos_lu = float(tf.read().strip())
                        tip_pos_norm = tip_pos_lu / nx
                else:
                    print(f"   [AVISO] tip_position.txt não encontrado para escala {k}")
                    tip_pos_norm = np.nan
            except Exception as e:
                print(f"Erro ao ler dados: {e}")
                tip_pos_norm = np.nan

            print(f"   [OK] Grid: {nx}x{ny} | Tip Position (x/L): {tip_pos_norm:.5f}")
            results.append({
                'scale': k,
                'nx': nx,
                'ny': ny,
                'dx': 1.0 / nx,
                'tip_position_normalized': tip_pos_norm
            })

    finally:
        if os.path.exists(BACKUP_PATH):
            shutil.move(BACKUP_PATH, CONFIG_PATH)
            print("\nConfiguração original restaurada.")

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"Resultados salvos em {RESULTS_CSV}")


if __name__ == "__main__":
    run_tests()