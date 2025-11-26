#include "Simulation.hpp"
#include "Config.hpp"
#include <random>
#include <omp.h>
#include <iostream>
#include <algorithm>

void generate_porous_media(LatticeMesh& mesh) {
    std::mt19937 rng(OBSTACLE_SEED);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // 1. Ruído
    for(int y = 0; y < NY; ++y) {
        for(int x = 0; x < NX; ++x) {
            if (x < 10 || x > NX - 10) {
                mesh.mask[mesh.idx(x, y)] = 0;
                continue;
            }
            if (dist(rng) > POROSITY_TARGET) mesh.mask[mesh.idx(x, y)] = 1;
            else mesh.mask[mesh.idx(x, y)] = 0;
        }
    }
    // 2. Autômato Celular (Suavização)
    std::vector<int> temp_mask = mesh.mask;
    for (int iter = 0; iter < 4; ++iter) {
        for(int y = 1; y < NY - 1; ++y) {
            for(int x = 10; x < NX - 10; ++x) {
                int neighbors = 0;
                for(int dy=-1; dy<=1; ++dy) {
                    for(int dx=-1; dx<=1; ++dx) {
                        if(dx==0 && dy==0) continue;
                        if(mesh.mask[mesh.idx(x+dx, y+dy)] == 1) neighbors++;
                    }
                }
                size_t id = mesh.idx(x, y);
                if (mesh.mask[id] == 1) temp_mask[id] = (neighbors >= 4)? 1 : 0;
                else temp_mask[id] = (neighbors >= 5)? 1 : 0;
            }
        }
        mesh.mask = temp_mask;
    }
    std::cout << "-> Meio poroso gerado." << std::endl;
}

void initialize(LatticeMesh& mesh) {
    double nu_oil = NU_CO2 * VISC_RATIO;
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> rdist(-0.02, 0.02);

    for(int y = 0; y < NY; ++y) {
        for(int x = 0; x < NX; ++x) {
            size_t id = mesh.idx(x, y);
            mesh.rho[id] = RHO0;
            mesh.ux[id] = 0.0; mesh.uy[id] = 0.0;
            mesh.C[id] = 0.0;
            mesh.nu_loc[id] = nu_oil;

            if (x == 0) {
                double perturb = 0.03 * std::sin(2.0*M_PI * y / (double)NY) + rdist(rng);
                double cval = 1.0 + perturb;
                if (cval > 1.0) cval = 1.0; if (cval < 0.0) cval = 0.0;
                mesh.C[id] = cval;
            }
            double u_sq = 0.0;
            for(int k=0; k<9; ++k) {
                double term = 1.0 - 1.5*u_sq;
                mesh.f[mesh.idx(x,y,k)] = W[k] * RHO0 * term;
                mesh.g[mesh.idx(x,y,k)] = W[k] * mesh.C[id] * term;
            }
        }
    }
}

void collide_and_stream(LatticeMesh& mesh) {
    double diff = (U_INLET * (double)NX) / PECLET;
    double tau_g = 0.5 + diff / CS_SQ;
    if (tau_g <= 0.5001) tau_g = 0.5001;
    double omega_g = 1.0 / tau_g;

    std::fill(mesh.f_new.begin(), mesh.f_new.end(), 0.0);
    std::fill(mesh.g_new.begin(), mesh.g_new.end(), 0.0);

    #pragma omp parallel for schedule(dynamic)
    for(int y = 0; y < NY; ++y) {
        for(int x = 0; x < NX; ++x) {
            size_t id_node = mesh.idx(x,y);
            if(mesh.mask[id_node] == 1) continue;

            double rho = 0.0, ux = 0.0, uy = 0.0, C = 0.0;
            for(int k=0; k<9; ++k) {
                double vf = mesh.f[mesh.idx(x,y,k)];
                rho += vf; ux += vf * CX[k]; uy += vf * CY[k];
                C += mesh.g[mesh.idx(x,y,k)];
            }
            if (rho > 1e-12) { ux /= rho; uy /= rho; }
            else { rho=RHO0; ux=0; uy=0; }

            mesh.rho[id_node] = rho; mesh.ux[id_node] = ux; mesh.uy[id_node] = uy;

            double C_phys = std::max(0.0, std::min(1.0, C));
            double nu_mix = (NU_CO2 * VISC_RATIO) * std::pow(NU_CO2 / (NU_CO2 * VISC_RATIO), C_phys);
            mesh.nu_loc[id_node] = nu_mix;

            double tau_plus = 0.5 + nu_mix / CS_SQ;
            if (tau_plus <= 0.5001) tau_plus = 0.5001;
            double tau_minus = 0.5 + MAGIC_LAMBDA / (tau_plus - 0.5);
            if (tau_minus <= 0.5001) tau_minus = tau_plus;
            
            double omega_plus = 1.0 / tau_plus;
            double omega_minus = 1.0 / tau_minus;
            double u_sq = ux*ux + uy*uy;

            for(int k=0; k<9; ++k) {
                int k_opp = OPP[k];
                double fk = mesh.f[mesh.idx(x,y,k)];
                double fk_opp = mesh.f[mesh.idx(x,y,k_opp)];
                double cu = CX[k]*ux + CY[k]*uy;

                double feq_sym = W[k] * rho * (1.0 + 4.5*cu*cu - 1.5*u_sq);
                double feq_asym = W[k] * rho * (3.0 * cu);
                
                double f_sym = 0.5 * (fk + fk_opp);
                double f_asym = 0.5 * (fk - fk_opp);
                double f_post = fk - omega_plus*(f_sym - feq_sym) - omega_minus*(f_asym - feq_asym);

                double geq_k = W[k] * C_phys * (1.0 + 3.0*cu);
                double gk = mesh.g[mesh.idx(x,y,k)];
                double g_post = gk * (1.0 - omega_g) + omega_g * geq_k;

                int nx = x + CX[k];
                int ny = y + CY[k];
                bool bounce = (nx < 0 || nx >= NX || ny < 0 || ny >= NY || mesh.mask[mesh.idx(nx, ny)] == 1);

                if (bounce) {
                    mesh.f_new[mesh.idx(x, y, k_opp)] += f_post;
                    mesh.g_new[mesh.idx(x, y, k_opp)] += g_post;
                } else {
                    mesh.f_new[mesh.idx(nx, ny, k)] += f_post;
                    mesh.g_new[mesh.idx(nx, ny, k)] += g_post;
                }
            }
        }
    }
}

void update_buffers(LatticeMesh& mesh) {
    mesh.f.swap(mesh.f_new);
    mesh.g.swap(mesh.g_new);
    std::fill(mesh.f_new.begin(), mesh.f_new.end(), 0.0);
    std::fill(mesh.g_new.begin(), mesh.g_new.end(), 0.0);
}

void update_macroscopics_C(LatticeMesh& mesh, double& total_mass) {
    total_mass = 0.0;
    #pragma omp parallel for reduction(+:total_mass)
    for(int y=0;y<NY;++y){
        for(int x=0;x<NX;++x){
            size_t id = mesh.idx(x,y);
            if (mesh.mask[id]==1) { mesh.C[id] = 0.0; continue; }
            double Cmac = 0.0;
            for(int k=0;k<9;++k) Cmac += mesh.g[mesh.idx(x,y,k)];
            if (Cmac < 0.0) Cmac = 0.0; if (Cmac > 1.0) Cmac = 1.0;
            mesh.C[id] = Cmac;
            total_mass += Cmac;
        }
    }
}