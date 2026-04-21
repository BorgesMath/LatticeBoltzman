# config.py
import numpy as np

# ==========================================
# 1. PARÂMETROS DE SIMULAÇÃO E TOPOLOGIA
# ==========================================
NY = 600
NX = 1800
MAX_ITER = 10000
SNAPSHOT_STEPS = 200

# ==========================================
# 2. HIDRODINÂMICA E CINEMÁTICA
# ==========================================
TAU_IN = 0.6   # AUMENTADO DE 0.52 PARA 0.65 (Segurança contra overflow)
TAU_OUT = 1.4
U_INLET = 0.05

# Permeabilidade Absoluta Basal (Lattice Units)
K_0 = 200.0 # 5000.0

# ==========================================
# 3. TERMODINÂMICA DE INTERFACE (CAHN-HILLIARD)
# ==========================================
M_MOBILITY = 0.05 # 0.05
SIGMA = 0.0001
INTERFACE_WIDTH = 3.0
CH_SUBSTEPS = 50
DT_CH = 1.0 / CH_SUBSTEPS

BETA = 3.0 * SIGMA * INTERFACE_WIDTH / 4.0
KAPPA = 3.0 * SIGMA * INTERFACE_WIDTH / 8.0

# ==========================================
# 4. MAGNETOSTÁTICA E CONTROLE DE SOLVER
# ==========================================
# Opções: 'POISSON' (Numérico) ou SEI LÁ
METODO_MAGNETISMO = 'POISSON'

H0 = 0 # 0.08
CHI_MAX = 1.2
SOR_OMEGA = 1.85

# Configurações para o Mtodo Analítico (Paramagnético)
H_ANGLE = 45  # 0.0 = Campo Vertical (Normal), 90.0 = Horizontal

# ==========================================
# 5. TENSORES DO MODELO LBM D2Q9
# ==========================================
W_LBM = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float64)
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)