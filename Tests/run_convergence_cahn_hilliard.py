# run_convergence_cahn_hilliard.py
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
# WORKER: ISOLA O ACOPLAMENTO CH-LBM PARA UMA GOTA ESTÁTICA
# ==============================================================================
if '--worker' in sys.argv:
    sys.path.insert(0, ROOT_DIR)
    from config.config import NY, NX, MAX_ITER, W_LBM, R_DROP, INTERFACE_WIDTH, SIGMA
    from lbm.lbm import lbm_step
    from cahn_hilliard.cahn_hilliard import cahn_hilliard_substep
    from tqdm import tqdm


    def run_worker():
        f = np.zeros((NY, NX, 9), dtype=np.float64)
        phi = np.zeros((NY, NX), dtype=np.float64)
        psi = np.zeros((NY, NX), dtype=np.float64)
        chi_field = np.zeros((NY, NX), dtype=np.float64)
        K_field = np.ones((NY, NX), dtype=np.float64) * 1e12
        rho = np.ones((NY, NX), dtype=np.float64)
        u_x = np.zeros((NY, NX), dtype=np.float64)
        u_y = np.zeros((NY, NX), dtype=np.float64)

        # 1. Alocação Analítica
        cx, cy = NX / 2.0, NY / 2.0
        for y in range(NY):
            for x in range(NX):
                r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                phi[y, x] = -np.tanh((r - R_DROP) / (INTERFACE_WIDTH / 2.0))

        # 2. Pré-Condicionamento Termodinâmico Extremo
        for _ in range(2500):
            phi = cahn_hilliard_substep(phi, u_x, u_y)

        for i in range(9):
            f[:, :, i] = W_LBM[i] * rho

        # 3. Integração Temporal LBM acoplado (Atingindo Regime Estacionário Capilar)
        for step in tqdm(range(MAX_ITER), desc=f"Laplace R={R_DROP}", disable=True):
            for _ in range(10):
                phi = cahn_hilliard_substep(phi, u_x, u_y)

            f, rho, u_x, u_y = lbm_step(f, phi, psi, rho, u_x, u_y, chi_field, K_field)

            # Blindagem de Fronteira
            f[:, 0, :] = f[:, 1, :]
            f[:, -1, :] = f[:, -2, :]
            rho[:, 0] = rho[:, 1]
            rho[:, -1] = rho[:, -2]

        # 4. Extração de Dados
        rho_in = rho[int(cy), int(cx)]
        # Ponto de referência mais isolado das quinas
        rho_out = rho[int(cy), 5]

        delta_p = np.abs((rho_in - rho_out) / 3.0)

        # Filtro Espacial: Ignora os nós de parede (quinas geram singularidades u_max irreais)
        u_max = np.max(np.sqrt(u_x[5:-5, 5:-5] ** 2 + u_y[5:-5, 5:-5] ** 2))

        temp_file = os.path.join(ROOT_DIR, "temp_laplace.txt")
        with open(temp_file, "w") as fp:
            fp.write(f"{delta_p},{u_max}")


    run_worker()
    sys.exit(0)


# ==============================================================================
# COORDENADOR: GERA OS RAIOS, EXECUTA E VALIDA A LEI DE LAPLACE
# ==============================================================================
def run_coordinator():
    radii = [12, 16, 20, 24, 28, 32]

    BASE_NY = 120
    BASE_NX = 120

    # ATUALIZAÇÃO CRÍTICA: Tempo de relaxação capilar exige ordem de 10^4 iterações
    # para equilibrar fluidos de alta densidade face a pequenas tensões superficiais.
    MAX_ITER = 15000

    SIGMA_TEST = 0.0001
    W_TEST = 5.0
    M_MOBILITY = 0.002

    if os.path.exists(CONFIG_PATH):
        shutil.copy(CONFIG_PATH, BACKUP_PATH)

    results = []

    try:
        for r in radii:
            content = f"""# config.py - GERADO PARA VALIDAÇÃO DE CAHN-HILLIARD (R={r})
import numpy as np

NY = {BASE_NY}
NX = {BASE_NX}
MAX_ITER = {MAX_ITER}
SNAPSHOT_STEPS = 0

TAU_IN = 1.0
TAU_OUT = 1.0
U_INLET = 0.0
K_0 = 1e12

M_MOBILITY = {M_MOBILITY}
SIGMA = {SIGMA_TEST}
INTERFACE_WIDTH = {W_TEST}
CH_SUBSTEPS = 10
DT_CH = 1.0 / CH_SUBSTEPS

BETA = (3.0 * SIGMA) / (4.0 * INTERFACE_WIDTH)
KAPPA = (3.0 * SIGMA * INTERFACE_WIDTH) / 8.0

METODO_MAGNETISMO = 'NENHUM'
H0 = 0.0
CHI_MAX = 0.0
SOR_OMEGA = 1.85 
H_ANGLE = 0.0
INITIAL_AMPLITUDE = 0.0
R_DROP = {r}

W_LBM = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float64)
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)
"""
            with open(CONFIG_PATH, 'w') as f:
                f.write(content)

            print(f"---> Validando Cahn-Hilliard (Gota Estática): Raio R = {r}")

            result = subprocess.run(
                [sys.executable, os.path.abspath(__file__), '--worker'],
                cwd=ROOT_DIR, capture_output=True, text=True, encoding='utf-8', errors='replace'
            )

            if result.returncode != 0:
                print(f"Erro no Kernel:\n{result.stderr}")
                continue

            temp_file = os.path.join(ROOT_DIR, "temp_laplace.txt")
            if os.path.exists(temp_file):
                with open(temp_file, "r") as fp:
                    data = fp.read().split(',')
                    dp = float(data[0])
                    u_max = float(data[1])

                inv_r = 1.0 / r
                results.append({'R': r, '1/R': inv_r, 'DeltaP': dp, 'U_Spurious': u_max})
                print(f"     [OK] Delta P: {dp:.6e} | Corrente Espúria Máx (Ma): {u_max:.2e}")
            else:
                print("     [ERRO] Falha na extração dos dados.")

    finally:
        if os.path.exists(BACKUP_PATH):
            shutil.move(BACKUP_PATH, CONFIG_PATH)
        temp_file = os.path.join(ROOT_DIR, "temp_laplace.txt")
        if os.path.exists(temp_file):
            os.remove(temp_file)

    if not results:
        return

    df = pd.DataFrame(results)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    inv_R_vals = df['1/R'].values
    dP_vals = df['DeltaP'].values

    m_num = np.linalg.lstsq(inv_R_vals[:, np.newaxis], dP_vals, rcond=None)[0][0]

    axes[0].plot(inv_R_vals, dP_vals, 'ko', markersize=8, label='LBM-CH Numérico')
    axes[0].plot(inv_R_vals, m_num * inv_R_vals, 'b-', linewidth=2,
                 label=fr'Ajuste Linear ($\sigma_{{num}} = {m_num:.6f}$)')
    axes[0].plot(inv_R_vals, SIGMA_TEST * inv_R_vals, 'r--', linewidth=2,
                 label=fr'Teórico ($\sigma_{{in}} = {SIGMA_TEST:.6f}$)')

    axes[0].set_xlabel(r'Curvatura ($1/R$)')
    axes[0].set_ylabel(r'Salto de Pressão ($|\Delta P|$)')
    axes[0].set_title('Validação da Lei de Laplace')
    axes[0].legend()

    axes[1].plot(df['R'], df['U_Spurious'], 'rs-', linewidth=2, markersize=8)
    axes[1].set_xlabel(r'Raio da Gota ($R$)')
    axes[1].set_ylabel(r'Magnitude da Velocidade Parasita ($u_{max}$)')
    axes[1].set_title('Análise de Correntes Espúrias')

    axes[1].set_yscale('log')
    axes[1].axhline(y=1e-3, color='k', linestyle=':', label='Limite Aceitável de Erro (Mach < 0.001)')
    axes[1].legend()

    plt.tight_layout()
    img_path = os.path.join(ROOT_DIR, 'validacao_cahn_hilliard.png')
    plt.savefig(img_path, dpi=300)
    print(f"\nAnálise concluída. Gráfico acadêmico salvo em: {img_path}")


if __name__ == "__main__":
    run_coordinator()