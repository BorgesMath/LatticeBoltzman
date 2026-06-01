# initialization/initialization.py
import numpy as np
from numba import njit, prange

# Tensores D2Q9
W_LBM = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36], dtype=np.float64)
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32)


@njit(parallel=True, cache=True)
def _init_kernel(phi, f, psi, rho, u_x, ny, nx, mode_m, amplitude, interface_width, x_center, Hx, Hy, u_inlet,
                 rho_base):
    u_sq = u_inlet ** 2

    for y in prange(ny):
        dist = x_center + amplitude * np.cos(2.0 * np.pi * mode_m * y / ny)

        for x in range(nx):
            phi[y, x] = -np.tanh((x - dist) / interface_width)
            psi[y, x] = 0.0  # Anulação para isolamento da perturbação
            u_x[y, x] = u_inlet

            # Injeção da pressão analítica pré-calculada
            rho_local = rho_base[x]
            rho[y, x] = rho_local

            for i in range(9):
                cu = CX[i] * u_inlet
                f[y, x, i] = W_LBM[i] * rho_local * (1.0 + 3.0 * cu + 4.5 * (cu ** 2) - 1.5 * u_sq)


def _campo_K_heterogeneo(ny, nx, K_0, corr_length, sigma_ln, seed=0):
    """
    Campo de permeabilidade log-normal correlacionado, média aritmética = K_0.
    "Homogêneo no geral, heterogêneo em pequenas seções": ruído branco gaussiano
    filtrado no espaço de Fourier por um núcleo gaussiano de comprimento de
    correlação `corr_length` (l.u.), periódico em x e y, depois exponenciado
    (log-normal). `g` é cortado em ±3 desvios para evitar K extremo (Forchheimer
    local em poros muito permeáveis / barreiras quase impermeáveis).

    Serve de gatilho de banda larga para a ramificação dos dedos (tip-splitting):
    perturba a interface em todos os comprimentos de onda e cria canais
    preferenciais que nucleiam e dividem os dedos. Roda 1× na inicialização
    (numpy puro; o kernel LBM já lê K_field[y,x] por nó).
    """
    rng = np.random.default_rng(seed)
    ruido = rng.standard_normal((ny, nx))
    kx = np.fft.fftfreq(nx).reshape(1, nx)
    ky = np.fft.fftfreq(ny).reshape(ny, 1)
    k2 = (2.0 * np.pi * kx) ** 2 + (2.0 * np.pi * ky) ** 2
    filtro = np.exp(-0.25 * k2 * corr_length ** 2)   # covariância gaussiana ~corr_length
    g = np.fft.ifft2(np.fft.fft2(ruido) * filtro).real
    g -= g.mean()
    std = g.std()
    if std > 1e-12:
        g /= std                                     # variância unitária
    g = np.clip(g, -3.0, 3.0)                         # corta caudas
    K = K_0 * np.exp(sigma_ln * g)
    K *= K_0 / K.mean()                               # média aritmética exata = K_0
    return K.astype(np.float64)


def initialize_fields(params):
    """
    inicializa os campos
    """
    ny, nx = params["NY"], params["NX"]
    u_inlet = params["U_INLET"]
    # Posição inicial da interface (l.u. a partir do inlet). Parametrizável via
    # casos.json ("X_CENTER"); default 80 preserva o comportamento histórico.
    # Valores maiores dão folga para a cavidade (trough) recuar sem bater no inlet.
    x_center = float(params.get("X_CENTER", 80.0))

    # =========================================================================
    # CÁLCULO ANALÍTICO DO FLUXO BASE DE DARCY (Evita decaimento da velocidade)
    # =========================================================================
    rho_base = np.ones(nx, dtype=np.float64)
    nu_in = (params["TAU_IN"] - 0.5) / 3.0
    nu_out = (params["TAU_OUT"] - 0.5) / 3.0

    dpdx_in = 3.0 * (nu_in / params["K_0"]) * u_inlet
    dpdx_out = 3.0 * (nu_out / params["K_0"]) * u_inlet

    # Integração cumulativa de Trás para Frente (Outlet ancorado em rho=1.0)
    for x in range(nx - 2, -1, -1):
        if x >= x_center:
            rho_base[x] = rho_base[x + 1] + dpdx_out
        else:
            rho_base[x] = rho_base[x + 1] + dpdx_in
    # =========================================================================

    f_a = np.empty((ny, nx, 9), dtype=np.float64)
    f_b = np.empty((ny, nx, 9), dtype=np.float64)
    phi_a = np.empty((ny, nx), dtype=np.float64)
    phi_b = np.empty((ny, nx), dtype=np.float64)

    psi = np.empty((ny, nx), dtype=np.float64)
    rho = np.empty((ny, nx), dtype=np.float64)
    u_x = np.empty((ny, nx), dtype=np.float64)
    u_y = np.zeros((ny, nx), dtype=np.float64)
    K_field = np.ones((ny, nx), dtype=np.float64) * params["K_0"]
    if params.get("K_HETEROGENEO", False):
        # Meio poroso heterogêneo (média = K_0): promove nucleação/ramificação
        # dos dedos. Ver casos_Ramificacao.json e _campo_K_heterogeneo acima.
        K_field = _campo_K_heterogeneo(
            ny, nx, params["K_0"],
            float(params.get("K_CORR_LENGTH", 40.0)),
            float(params.get("K_SIGMA_LN", 0.5)),
            int(params.get("K_SEED", 0)))

    Fx = np.zeros((ny, nx), dtype=np.float64)
    Fy = np.zeros((ny, nx), dtype=np.float64)
    mu_buffer = np.zeros((ny, nx), dtype=np.float64)

    angle_rad = np.radians(params["H_ANGLE"])
    Hx = params["H0"] * np.cos(angle_rad)
    Hy = params["H0"] * np.sin(angle_rad)

    # Passamos o rho e o rho_base para o kernel
    _init_kernel(phi_a, f_a, psi, rho, u_x, ny, nx, params["mode_m"],
                 params["amplitude"], params["INTERFACE_WIDTH"], x_center, Hx, Hy, u_inlet, rho_base)

    f_b[:] = f_a[:]
    phi_b[:] = phi_a[:]

    return (f_a, f_b), (phi_a, phi_b), psi, rho, u_x, u_y, K_field, (Fx, Fy, mu_buffer)