#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <cmath>

// ============================================================================
// PARÂMETROS DA SIMULAÇÃO
// ============================================================================
constexpr int NX = 400;             
constexpr int NY = 150;             
constexpr int MAX_ITER = 30000;     
constexpr int OUTPUT_FREQ = 200;    

// Parâmetros Físicos 
constexpr double RE_COARSE = 0.2;   
constexpr double VISC_RATIO = 50.0; 
constexpr double PECLET = 100.0;    

// Propriedades do Fluido
constexpr double NU_CO2 = 0.005;    
constexpr double RHO0 = 1.0;        

// Parâmetros LBM
constexpr double CS_SQ = 1.0 / 3.0; 
constexpr double MAGIC_LAMBDA = 0.25;
constexpr double U_INLET = 0.02;    

// Meio Poroso
constexpr double POROSITY_TARGET = 0.75; 
constexpr int OBSTACLE_SEED = 42;        

// ============================================================================
// CONSTANTES DO LATTICE D2Q9
// ============================================================================
const double W[9] = { 4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36. };
const int CX[9]   = {0, 1, 0, -1, 0,  1, -1, -1,  1};
const int CY[9]   = {0, 0, 1,  0, -1, 1,  1, -1, -1};
const int OPP[9]  = {0, 3, 4,  1,  2, 7,  8,  5,  6};

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#endif