# run_convergence_lbm_core.py
import os
import sys
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, 'config', 'config.py')
BACKUP_PATH = os.path.join(ROOT_DIR, 'config', 'config_backup.py')

# ==============================================================================
# WORKER: ISOLA O LBM_STEP E AVALIA A FORMA DA PARÁBOLA
# ==============================================================================
if '--worker' in sys.argv:
    sys.path.insert(0, ROOT_DIR)
    from config.config import NY, NX, MAX_ITER, U_INLET, W_LBM
    from lbm.lbm import lbm_step
    from tqdm import tqdm


    def run_worker():
        f = np.zeros((NY, NX, 9), dtype=np.float64)
        phi = np.ones((NY, NX), dtype=np.float64)
        psi = np.zeros((NY, NX), dtype=np.float64)
        chi_field = np.zeros((NY, NX), dtype=np.float64)
        K_field = np.ones((NY, NX), dtype=np.float64) * 1e12  # Desliga Meio Poroso
        rho = np.ones((NY, NX), dtype=np.float64)
        u_x = np.zeros((NY, NX), dtype=np.float64)
        u_y = np.zeros((NY, NX), dtype=np.float64)

        for i in range(9):
            f[:, :, i] = W_LBM[i] * rho

        # TQDM desativado para evitar UnicodeDecodeError no buffer do Windows
        for step in tqdm(range(MAX_ITER), desc=f"LBM Posiueille NX={NX}", disable=True):
            f, rho, u_x, u_y = lbm_step(f, phi, psi, rho, u_x, u_y, chi_field, K_field)

        # Extração
        coluna_extracao = NX - 5
        u_prof = u_x[:, coluna_extracao]

        # -----------------------------------------------------
        # CÁLCULO ANALÍTICO (Forma Parabólica Normalizada)
        # -----------------------------------------------------
        H = NY
        yc = (H - 1) / 2.0
        y_coords = np.arange(H)

        # Parábola adimensional teórica (Máximo rigorosamente = 1.0)
        u_ana_norm = 1.0 - ((y_coords - yc) / (H / 2.0)) ** 2

        # Normalização do Perfil Numérico (Isola o erro de cisalhamento/malha)
        max_num = np.max(u_prof)
        if max_num > 1e-8:
            u_prof_norm = u_prof / max_num
        else:
            u_prof_norm = u_prof

        # Erro L2 sobre a GEOMETRIA da curva
        error_l2 = np.sqrt(np.sum((u_prof_norm - u_ana_norm) ** 2)) / np.sqrt(np.sum(u_ana_norm ** 2))

        temp_file = os.path.join(ROOT_DIR, "temp_error.txt")
        with open(temp_file, "w") as fp:
            fp.write(str(error_l2))


    run_worker()
    sys.exit(0)


# ==============================================================================
# COORDENADOR: ESCALONAMENTO E GRÁFICO
# ==============================================================================
def run_coordinator():
    scales = [1, 2, 4, 8]
    BASE_NY = 15
    BASE_NX = 60
    BASE_U = 0.05
    BASE_ITER = 2500

    if os.path.exists(CONFIG_PATH):
        shutil.copy(CONFIG_PATH, BACKUP_PATH)

    results = []

    try:
        for k in scales:
            ny = BASE_NY * k
            nx = BASE_NX * k
            u_in = BASE_U / k
            max_iter = BASE_ITER * (k ** 2)

            content = f"""# config.py - GERADO PARA ISOLAMENTO LBM (k={k})
import numpy as np

NY = {ny}
NX = {nx}
MAX_ITER = {max_iter}
SNAPSHOT_STEPS = 0

TAU_IN = 1.0
TAU_OUT = 1.0
U_INLET = {u_in}
K_0 = 1e12

# Desliga módulos acoplados
M_MOBILITY = 0.0
SIGMA = 0.0
INTERFACE_WIDTH = 3.0
CH_SUBSTEPS = 1
DT_CH = 1.0
BETA = 0.0
KAPPA = 0.0
METODO_MAGNETISMO = 'NENHUM'
H0 = 0.0
CHI_MAX = 0.0
SOR_OMEGA = 1.85 
H_ANGLE = 0.0
INITIAL_AMPLITUDE = 0.0

W_LBM = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float64)
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)
"""
            with open(CONFIG_PATH, 'w') as f:
                f.write(content)

            print(f"\n---> Validando Kernel LBM (Validação de Forma): Escala k={k} (Grid: {nx}x{ny})")

            result = subprocess.run(
                [sys.executable, os.path.abspath(__file__), '--worker'],
                cwd=ROOT_DIR, capture_output=True, text=True, encoding='utf-8', errors='replace'
            )

            if result.returncode != 0:
                print(f"Erro no Kernel:\n{result.stderr}")
                continue

            temp_file = os.path.join(ROOT_DIR, "temp_error.txt")
            with open(temp_file, "r") as fp:
                err = float(fp.read())

            dx_relativo = 1.0 / ny
            results.append({'k': k, 'dx': dx_relativo, 'erro': err})
            print(f"     [OK] Erro L2 (Forma da Parábola): {err:.6e}")

    finally:
        if os.path.exists(BACKUP_PATH):
            shutil.move(BACKUP_PATH, CONFIG_PATH)
        temp_file = os.path.join(ROOT_DIR, "temp_error.txt")
        if os.path.exists(temp_file):
            os.remove(temp_file)

    if not results:
        return

    df = pd.DataFrame(results)
    dx = df['dx'].values
    erros = df['erro'].values

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 6))

    plt.loglog(dx, erros, 'bo-', linewidth=2, markersize=8, label='LBM Numérico (Seu lbm.py)')

    C = erros[0] / (dx[0] ** 2)
    ref_order2 = C * (dx ** 2)
    plt.loglog(dx, ref_order2, 'k--', linewidth=1.5, label=r'Teórico $\mathcal{O}(\Delta x^2)$')

    plt.xlabel(r'Espaçamento da Malha ($\Delta x$)')
    plt.ylabel(r'Norma de Erro Relativo $L_2$ (Perfil Normalizado)')
    plt.title('Validação da Dinâmica Viscosa (LBM Isolado)')
    plt.legend()
    plt.tight_layout()

    img_path = os.path.join(ROOT_DIR, 'convergencia_LBM_core.png')
    plt.savefig(img_path, dpi=300)
    print(f"\nAnálise concluída. Gráfico acadêmico salvo em: {img_path}")


if __name__ == "__main__":
    run_coordinator()