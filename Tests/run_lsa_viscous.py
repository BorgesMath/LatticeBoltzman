# run_lsa_viscous.py
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

if '--worker' in sys.argv:
    sys.path.insert(0, ROOT_DIR)
    from config.config import NY, NX, MAX_ITER, W_LBM, INTERFACE_WIDTH, U_INLET
    from config.config import INITIAL_AMPLITUDE, CX, CY, TAU_IN, TAU_OUT
    from lbm.lbm import lbm_step
    from Multifasico.cahn_hilliard import cahn_hilliard_substep

    mode_m = int(sys.argv[2])


    def get_amplitude_subpixel(phi):
        ny_shape, nx_shape = phi.shape
        x_tips = []
        for y in range(ny_shape):
            crossings = np.where(np.diff(np.sign(phi[y, :])))[0]
            if len(crossings) > 0:
                x_idx = crossings[-1]
                phi0 = phi[y, x_idx]
                phi1 = phi[y, x_idx + 1]
                if np.abs(phi1 - phi0) > 1e-6:
                    x_real = x_idx - phi0 / (phi1 - phi0)
                    x_tips.append(x_real)
        if len(x_tips) > 0:
            return (np.max(x_tips) - np.min(x_tips)) / 2.0
        return 0.0


    def run_worker():
        f = np.zeros((NY, NX, 9), dtype=np.float64)
        phi = np.zeros((NY, NX), dtype=np.float64)
        psi = np.zeros((NY, NX), dtype=np.float64)
        chi_field = np.zeros((NY, NX), dtype=np.float64)

        from config.config import K_0 as K_VALUE
        K_field = np.ones((NY, NX), dtype=np.float64) * K_VALUE
        rho = np.ones((NY, NX), dtype=np.float64)

        # 1. Alocação da onda no centro do canal
        x_center = NX * 0.4
        for y in range(NY):
            dist = x_center + INITIAL_AMPLITUDE * np.sin(2.0 * np.pi * mode_m * y / NY)
            for x in range(NX):
                phi[y, x] = -np.tanh((x - dist) / (INTERFACE_WIDTH / 2.0))

        # 2. Relaxamento químico
        u_temp_x = np.zeros((NY, NX), dtype=np.float64)
        u_temp_y = np.zeros((NY, NX), dtype=np.float64)
        for _ in range(200):
            phi = cahn_hilliard_substep(phi, u_temp_x, u_temp_y)

        # 3. Equilíbrio de Pressão Darcy
        nu_in = (TAU_IN - 0.5) / 3.0
        nu_out = (TAU_OUT - 0.5) / 3.0

        for x in range(NX - 1, -1, -1):
            if x >= x_center:
                rho_val = 1.0 + 3.0 * (nu_out / K_VALUE) * U_INLET * (NX - 1 - x)
            else:
                rho_center = 1.0 + 3.0 * (nu_out / K_VALUE) * U_INLET * (NX - 1 - x_center)
                rho_val = rho_center + 3.0 * (nu_in / K_VALUE) * U_INLET * (x_center - x)
            rho[:, x] = rho_val

        u_x = np.ones((NY, NX), dtype=np.float64) * U_INLET
        u_y = np.zeros((NY, NX), dtype=np.float64)

        u_sq = u_x ** 2 + u_y ** 2
        for y in range(NY):
            for x in range(NX):
                for i in range(9):
                    cu = CX[i] * u_x[y, x] + CY[i] * u_y[y, x]
                    f[y, x, i] = W_LBM[i] * rho[y, x] * (1.0 + 3.0 * cu + 4.5 * cu ** 2 - 1.5 * u_sq[y, x])

        history_t = []
        history_A = []
        sample_rate = 20

        for step in range(MAX_ITER):
            for _ in range(10):
                phi = cahn_hilliard_substep(phi, u_x, u_y)
            f, rho, u_x, u_y = lbm_step(f, phi, psi, rho, u_x, u_y, chi_field, K_field)

            if step % sample_rate == 0:
                A_t = get_amplitude_subpixel(phi)
                history_t.append(step)
                history_A.append(A_t)

        t_vals = np.array(history_t)
        a_vals = np.array(history_A)

        # 4. ISOLAMENTO ESTRITO DO REGIME LINEAR (Filtro de Janela)
        # Descarta os primeiros 20% (Inércia acústica) e os últimos 20% (Saturação não-linear)
        start_idx = int(len(t_vals) * 0.2)
        end_idx = int(len(t_vals) * 0.8)

        t_fit = t_vals[start_idx:end_idx]
        ln_A = np.log(a_vals[start_idx:end_idx])

        omega_num = np.polyfit(t_fit, ln_A, 1)[0]

        temp_file = os.path.join(ROOT_DIR, f"temp_lsa_m{mode_m}.txt")
        with open(temp_file, "w") as fp:
            fp.write(str(omega_num))


    run_worker()
    sys.exit(0)


def run_coordinator():
    modos = [1, 2, 3, 4, 5, 6, 7]

    BASE_NY = 200
    BASE_NX = 600

    # Reduzido para focar exclusivamente na fase linear embrionária
    MAX_ITER = 1500

    # POTÊNCIA HIDRODINÂMICA MÁXIMA
    TAU_IN = 0.6  # Fluido de invasão ultra-fino
    TAU_OUT = 2.0  # Fluido residente ultra-viscoso
    U_IN = 0.05  # Injeção agressiva

    K_0 = 100.0  # Balanceamento Darcy-Brinkman
    SIGMA = 0.005  # Tensão superficial para cravamento do espectro
    M_MOB = 0.005

    A0 = 3.0  # Perturbação inicial visível

    if os.path.exists(CONFIG_PATH):
        shutil.copy(CONFIG_PATH, BACKUP_PATH)

    results = []

    try:
        for m in modos:
            content = f"""# config.py - GERADO PARA LSA (Modo {m})
import numpy as np

NY = {BASE_NY}
NX = {BASE_NX}
MAX_ITER = {MAX_ITER}
SNAPSHOT_STEPS = 0

TAU_IN = {TAU_IN}
TAU_OUT = {TAU_OUT}
U_INLET = {U_IN}
K_0 = {K_0}

M_MOBILITY = {M_MOB}
SIGMA = {SIGMA}
INTERFACE_WIDTH = 3.0
CH_SUBSTEPS = 10
DT_CH = 1.0 / CH_SUBSTEPS

BETA = (3.0 * SIGMA) / (4.0 * INTERFACE_WIDTH)
KAPPA = (3.0 * SIGMA * INTERFACE_WIDTH) / 8.0

METODO_MAGNETISMO = 'NENHUM'
H0 = 0.0
CHI_MAX = 0.0
SOR_OMEGA = 1.85 
H_ANGLE = 0.0
INITIAL_AMPLITUDE = {A0}

W_LBM = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float64)
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)
"""
            with open(CONFIG_PATH, 'w') as f:
                f.write(content)

            print(f"---> Extraindo Espectro LSA: Modo n = {m}")

            result = subprocess.run(
                [sys.executable, os.path.abspath(__file__), '--worker', str(m)],
                cwd=ROOT_DIR, capture_output=True, text=True, encoding='utf-8', errors='replace'
            )

            if result.returncode != 0:
                print(f"Erro no Kernel:\n{result.stderr}")
                continue

            temp_file = os.path.join(ROOT_DIR, f"temp_lsa_m{m}.txt")
            if os.path.exists(temp_file):
                with open(temp_file, "r") as fp:
                    omega_num = float(fp.read().strip())

                k_val = 2.0 * np.pi * m / BASE_NY
                results.append({'Modo': m, 'k': k_val, 'Omega_Num': omega_num})
                print(f"     [OK] k: {k_val:.4f} | Taxa Numérica: {omega_num:.6e}")
                os.remove(temp_file)

    finally:
        if os.path.exists(BACKUP_PATH):
            shutil.move(BACKUP_PATH, CONFIG_PATH)

    if not results: return

    df = pd.DataFrame(results)
    k_num = df['k'].values
    omega_num = df['Omega_Num'].values

    k_ana = np.linspace(0, np.max(k_num) * 1.1, 200)
    nu_in = (TAU_IN - 0.5) / 3.0
    nu_out = (TAU_OUT - 0.5) / 3.0

    # Relação Analítica
    omega_ana = (k_ana / (nu_in + nu_out)) * (U_IN * (nu_out - nu_in) - SIGMA * K_0 * (k_ana ** 2))

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(9, 6))

    plt.plot(k_ana, omega_ana, 'k-', linewidth=2, label='Teórico (Darcy Clássico)')
    plt.plot(k_num, omega_num, 'ro', markersize=8, label='LBM Numérico (Brinkman-CH)')

    plt.axhline(0, color='black', linewidth=1, linestyle='--')

    k_max = k_ana[np.argmax(omega_ana)]
    omega_max = np.max(omega_ana)
    plt.plot(k_max, omega_max, 'b*', markersize=12, label=rf'Crescimento Máximo Teórico ($k_c={k_max:.3f}$)')

    plt.xlabel(r'Número de Onda ($k$)')
    plt.ylabel(r'Taxa de Crescimento ($\omega$)')
    plt.title('Curva de Dispersão Linear (Calibração de Regime)')
    plt.legend()
    plt.tight_layout()

    img_path = os.path.join(ROOT_DIR, 'validacao_lsa_viscosa.png')
    plt.savefig(img_path, dpi=300)
    print(f"\nGráfico acadêmico corrigido salvo em: {img_path}")


if __name__ == "__main__":
    run_coordinator()