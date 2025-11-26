#include "Boundary.hpp"
#include "Config.hpp"
#include <omp.h>
#include <cmath>
#include <algorithm>

void apply_boundary_conditions(LatticeMesh& mesh) {
    #pragma omp parallel for
    for(int y = 0; y < NY; ++y) {
        if (mesh.mask[mesh.idx(0,y)] == 1) continue;

        // --- Inlet (Zou-He Velocity) ---
        double f0 = mesh.f_new[mesh.idx(0,y,0)];
        double f2 = mesh.f_new[mesh.idx(0,y,2)];
        double f4 = mesh.f_new[mesh.idx(0,y,4)];
        double f3 = mesh.f_new[mesh.idx(0,y,3)];
        double f6 = mesh.f_new[mesh.idx(0,y,6)];
        double f7 = mesh.f_new[mesh.idx(0,y,7)];

        double rho_in_denom = (1.0 - U_INLET);
        if (std::abs(rho_in_denom) < 1e-8) rho_in_denom = 1e-8;
        double rho_in = (f0 + f2 + f4 + 2.0*(f3 + f6 + f7)) / rho_in_denom;

        double f1 = f3 + (2.0/3.0)*rho_in*U_INLET;
        double f5 = f7 - 0.5*(f2 - f4) + (1.0/6.0)*rho_in*U_INLET;
        double f8 = f6 + 0.5*(f2 - f4) + (1.0/6.0)*rho_in*U_INLET;

        mesh.f_new[mesh.idx(0,y,1)] = f1;
        mesh.f_new[mesh.idx(0,y,5)] = f5;
        mesh.f_new[mesh.idx(0,y,8)] = f8;

        // --- Inlet Concentração ---
        double perturb = 0.02 * std::sin(2.0*M_PI * y / (double)NY);
        double C_in = 1.0 + perturb;
        if (C_in < 0.0) C_in = 0.0;
        if (C_in > 1.0) C_in = 1.0;

        for(int k = 0; k < 9; ++k) {
           double cu = CX[k]*U_INLET;
           double geq = W[k] * C_in * (1.0 + 3.0*cu);
           mesh.g_new[mesh.idx(0,y,k)] = geq;
        }
        mesh.C[mesh.idx(0,y)] = C_in;

        // --- Outlet (Extrapolação simples - Neumann nulo) ---
        if (mesh.mask[mesh.idx(NX-1,y)] == 0) {
            for(int k=0; k<9; ++k) {
                mesh.f_new[mesh.idx(NX-1,y,k)] = mesh.f_new[mesh.idx(NX-2,y,k)];
                mesh.g_new[mesh.idx(NX-1,y,k)] = mesh.g_new[mesh.idx(NX-2,y,k)];
            }
        }
    }
}