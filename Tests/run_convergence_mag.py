# run_convergence_mag.py
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
# WORKER: VALIDAÇÃO COM SUSCETIBILIDADE LINEAR (SEM ERRO DE INTERPOLAÇÃO)
# ==============================================================================
if '--worker' in sys.argv:
    sys.path.insert(0, ROOT_DIR)
    from config.config import NY, NX
    from Magnetismo.poisson import solve_poisson_magnetic


    def run_worker():
        psi = np.zeros((NY, NX), dtype=np.float64)
        chi_field = np.zeros((NY, NX), dtype=np.float64)

        L = 1.0
        # Mapeamento estrito: x=0 é o primeiro nó, x=L é o último nó.
        dx = L / (NX - 1)

        # 1. Alocação do Campo Linear (mu = 1 + x)
        for y in range(NY):
            for x in range(NX):
                x_phys = x * dx
                chi_field[y, x] = x_phys  # Suscetibilidade Linear
                psi[y, x] = -x_phys  # Chute inicial

        # 2. Fronteiras Analíticas Exatas de Dirichlet
        # psi_ana(x) = -ln(1 + x)
        for y in range(NY):
            psi[y, 0] = -np.log(1.0 + 0.0)
            psi[y, NX - 1] = -np.log(1.0 + L)

        # 3. Iteração Contínua (Tolerância Extrema e Estabilidade Absoluta)
        tol = 1e-12
        max_calls = 10000  # Garante até 150.000 sweeps na malha
        converged = False

        for call in range(max_calls):
            psi_old = psi.copy()
            psi = solve_poisson_magnetic(psi, chi_field)

            err = np.max(np.abs(psi - psi_old))
            if err < tol:
                converged = True
                break

        # 4. Extração Numérica no Eixo Central
        y_center = int(NY / 2)
        Hx_num = np.zeros(NX)
        Hx_ana = np.zeros(NX)
        x_coords = np.zeros(NX)

        for x in range(1, NX - 1):
            x_phys = x * dx
            x_coords[x] = x_phys

            # Numérico: Diferença Central
            Hx_num[x] = -(psi[y_center, x + 1] - psi[y_center, x - 1]) / (2.0 * dx)

            # Analítico Exato para este PVI
            Hx_ana[x] = 1.0 / (1.0 + x_phys)

        # Remove as fronteiras de Dirichlet da métrica de erro
        pad = 2
        H_n_core = Hx_num[pad:-pad]
        H_a_core = Hx_ana[pad:-pad]

        err_l2 = np.linalg.norm(H_n_core - H_a_core) / np.linalg.norm(H_a_core)

        temp_file = os.path.join(ROOT_DIR, "temp_mag.txt")
        with open(temp_file, "w") as fp:
            fp.write(f"{err_l2}\n")
            fp.write(",".join(map(str, x_coords[pad:-pad])) + "\n")
            fp.write(",".join(map(str, H_n_core)) + "\n")
            fp.write(",".join(map(str, H_a_core)))


    run_worker()
    sys.exit(0)


# ==============================================================================
# COORDENADOR: GERA MALHAS, EXECUTA O WORKER E PLOTA OS RESULTADOS
# ==============================================================================
def run_coordinator():
    # Resolucões dimensionadas para solvers diretos garantirem convergência
    scales = [1, 2, 4, 8]
    BASE_NY = 20
    BASE_NX = 20

    if os.path.exists(CONFIG_PATH):
        shutil.copy(CONFIG_PATH, BACKUP_PATH)

    results = []
    best_X, best_H_num, best_H_ana = [], [], []

    try:
        for k in scales:
            ny = BASE_NY * k
            nx = BASE_NX * k

            content = f"""# config.py - GERADO PARA VALIDAÇÃO DE POISSON
import numpy as np

NY = {ny}
NX = {nx}
MAX_ITER = 1
SNAPSHOT_STEPS = 0
TAU_IN = 1.0
TAU_OUT = 1.0
U_INLET = 0.0
K_0 = 1e12
M_MOBILITY = 0.0
SIGMA = 0.0
INTERFACE_WIDTH = 3.0
CH_SUBSTEPS = 1
DT_CH = 1.0
BETA = 0.0
KAPPA = 0.0
METODO_MAGNETISMO = 'POISSON'
H0 = 1.0
CHI_MAX = 0.0

# Gauss-Seidel Puro: Ligeiramente mais lento, mas impossível de divergir/oscilar.
SOR_OMEGA = 1.0 
H_ANGLE = 0.0
INITIAL_AMPLITUDE = 0.0

W_LBM = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float64)
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)
"""
            with open(CONFIG_PATH, 'w') as f:
                f.write(content)

            print(f"---> Validando Solver poisson.py: Escala k={k} (Grid: {nx}x{ny})")

            result = subprocess.run(
                [sys.executable, os.path.abspath(__file__), '--worker'],
                cwd=ROOT_DIR, capture_output=True, text=True, encoding='utf-8', errors='replace'
            )

            if result.returncode != 0:
                print(f"Erro no Kernel:\n{result.stderr}")
                continue

            temp_file = os.path.join(ROOT_DIR, "temp_mag.txt")
            if os.path.exists(temp_file):
                with open(temp_file, "r") as fp:
                    lines = fp.readlines()
                    err = float(lines[0].strip())

                    if k == scales[-1]:
                        best_X = np.array([float(x) for x in lines[1].strip().split(',')])
                        best_H_num = np.array([float(x) for x in lines[2].strip().split(',')])
                        best_H_ana = np.array([float(x) for x in lines[3].strip().split(',')])

                # Métrica quadrática rigorosa
                results.append({'k': k, 'dx': 1.0 / (nx - 1), 'erro': err})
                print(f"     [OK] Erro L2: {err:.6e}")

    finally:
        if os.path.exists(BACKUP_PATH):
            shutil.move(BACKUP_PATH, CONFIG_PATH)
        temp_file = os.path.join(ROOT_DIR, "temp_mag.txt")
        if os.path.exists(temp_file):
            os.remove(temp_file)

    if not results: return

    df = pd.DataFrame(results)
    dx_vals = df['dx'].values
    erros = df['erro'].values

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].loglog(dx_vals, erros, 'ro-', linewidth=2, markersize=8, label='Solver Numérico (Seu poisson.py)')
    C = erros[0] / (dx_vals[0] ** 2)
    axes[0].loglog(dx_vals, C * (dx_vals ** 2), 'k--', linewidth=1.5, label=r'Teórico $\mathcal{O}(\Delta x^2)$')
    axes[0].set_xlabel(r'Espaçamento da Malha ($\Delta x$)')
    axes[0].set_ylabel(r'Norma de Erro Relativo $L_2 (H_x)$')
    axes[0].set_title('Convergência do Operador de Poisson (Suscetibilidade Linear)')
    axes[0].legend()

    axes[1].plot(best_X, best_H_ana, 'k-', linewidth=3, label='Analítico Exato')
    axes[1].plot(best_X, best_H_num, 'r--', linewidth=2, label=f'Numérico LBM (k={scales[-1]})')
    axes[1].set_xlabel(r'Posição Transversal na Malha Física ($x$)')
    axes[1].set_ylabel(r'Intensidade do Campo ($H_x$)')
    axes[1].set_title(r'Validação da Distorção Magnética Contínua')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('validacao_magnetismo.png', dpi=300)
    print("\nGráfico acadêmico salvo em: validacao_magnetismo.png")


if __name__ == "__main__":
    run_coordinator()