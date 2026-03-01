# config.py
import numpy as np

# ==========================================
# 1. PARÂMETROS DE SIMULAÇÃO E TOPOLOGIA
# ==========================================
NY = 400
NX = 800
MAX_ITER = 8000
SNAPSHOT_STEPS = 15

PERIODIC_Y = False

# ==========================================
# 2. HIDRODINÂMICA E CINEMÁTICA
# ==========================================
TAU_IN = 0.52   # Fluido Invasor (Baixa viscosidade)
TAU_OUT = 3.5   # Fluido Residente (Alta viscosidade -> M ~ 125)
U_INLET = 0.02  # Velocidade de avanço na fronteira de entrada

# Permeabilidade Absoluta Basal (Lattice Units)
K_0 = 1500.0

# ==========================================
# 3. TERMODINÂMICA DE INTERFACE (CAHN-HILLIARD)
# ==========================================
M_MOBILITY = 0.002
SIGMA = 0.0001
INTERFACE_WIDTH = 3.0
CH_SUBSTEPS = 10
DT_CH = 1.0 / CH_SUBSTEPS

BETA =  0.001   #3.0 * SIGMA * INTERFACE_WIDTH / 4.0
KAPPA =  0.016 #3.0 * SIGMA * INTERFACE_WIDTH / 8.0

# ==========================================
# 4. MAGNETOSTÁTICA
# ==========================================
H0 = 0.05
CHI_MAX = 0.80
SOR_OMEGA = 1.85 # Fator de sobre-relaxação (Otimizado para malha 600x300)

# ==========================================
# 5. TENSORES DO MODELO LBM D2Q9
# ==========================================
W_LBM = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float64)
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)