#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <cmath>

// ============================================================================
// ESCOLHA DO MODO DE SIMULAÇÃO
// ============================================================================
enum SimType { MISCIBLE, IMMISCIBLE };

// MUDE AQUI PARA ALTERNAR O MODO:
constexpr SimType SIM_MODE = MISCIBLE; 
// MISCIBLE   = Advecção-Difusão (sua versão anterior)
// IMMISCIBLE = Shan-Chen Multicomponente (separação de fases)

// ============================================================================
// PARÂMETROS GERAIS
// ============================================================================
constexpr int NX = 400;             
constexpr int NY = 150;             
constexpr int MAX_ITER = 30000;     
constexpr int OUTPUT_FREQ = 200;    

// Parâmetros Físicos Gerais
constexpr double RHO0 = 1.0;        
constexpr double U_INLET = 0.05;    // Aumentei um pouco
constexpr double POROSITY_TARGET = 0.75; 
constexpr int OBSTACLE_SEED = 42;        

// ============================================================================
// PARÂMETROS ESPECÍFICOS: MISCÍVEL (Advecção-Difusão)
// ============================================================================
constexpr double PECLET = 100.0;    
constexpr double VISC_RATIO_MISC = 50.0; 
constexpr double NU_BASE = 0.02;    

// ============================================================================
// PARÂMETROS ESPECÍFICOS: IMISCÍVEL (Shan-Chen)
// ============================================================================
// G_INT controla a Tensão Superficial. 
// Para D2Q9, G_INT > 0 gera repulsão. Valores típicos: 1.0 a 3.0 (depende do rho)
// Se G for muito alto (>3.5), a simulação explode.
constexpr double G_INT = 1.2; 

// Viscosidade única para Shan-Chen simples (tau = 1.0 é bem estável)
constexpr double TAU_SC = 1.0; 

// Densidades iniciais para separação
constexpr double RHO_RED_INIT = 1.0;  // Fluido 1
constexpr double RHO_BLUE_INIT = 1.0; // Fluido 2

// ============================================================================
// CONSTANTES DO LATTICE D2Q9
// ============================================================================
constexpr double CS_SQ = 1.0 / 3.0; 
const double W[9] = { 4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36. };
const int CX[9]   = {0, 1, 0, -1, 0,  1, -1, -1,  1};
const int CY[9]   = {0, 0, 1,  0, -1, 1,  1, -1, -1};
const int OPP[9]  = {0, 3, 4,  1,  2, 7,  8,  5,  6};

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#endif