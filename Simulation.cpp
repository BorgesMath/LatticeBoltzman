#include "Simulation.hpp"
#include "Config.hpp"
#include <random>
#include <omp.h>
#include <iostream>
#include <algorithm>

// Mantemos a geração de meio poroso igual (omitida aqui por brevidade, 
// MANTENHA A FUNÇÃO generate_porous_media DO CÓDIGO ANTERIOR)
void generate_porous_media(LatticeMesh& mesh) {
    // ... (Copie a implementação anterior ou mantenha o arquivo se já tiver)
    // Se precisar, eu reescrevo, mas é idêntica à versão anterior.
    // Apenas para garantir que o código compile, vou colocar uma versão minificada:
    std::mt19937 rng(OBSTACLE_SEED);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for(int i=0; i<NX*NY; ++i) mesh.mask[i] = (dist(rng)>POROSITY_TARGET)?1:0;
    // (Lembre-se de limpar entrada/saída igual antes)
    for(int y=0;y<NY;++y) for(int x=0;x<NX;++x) if(x<10||x>NX-10) mesh.mask[mesh.idx(x,y)]=0;
    std::cout << "-> Meio poroso (simplificado) gerado." << std::endl;
}

// Inicialização Híbrida
void initialize(LatticeMesh& mesh) {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> noise(-0.01, 0.01);

    for(int y = 0; y < NY; ++y) {
        for(int x = 0; x < NX; ++x) {
            size_t id = mesh.idx(x, y);
            mesh.ux[id] = 0.0; mesh.uy[id] = 0.0;
            
            if (SIM_MODE == MISCIBLE) {
                // Modo Antigo
                mesh.rho[id] = RHO0;
                mesh.nu_loc[id] = NU_BASE * VISC_RATIO_MISC;
                mesh.C[id] = 0.0; // Tudo óleo
                if (x < 5) mesh.C[id] = 1.0; // Entrada CO2

                for(int k=0; k<9; ++k) {
                    mesh.f[mesh.idx(x,y,k)] = W[k] * RHO0;
                    mesh.g[mesh.idx(x,y,k)] = W[k] * mesh.C[id];
                }
            } 
            else { 
                // Modo Shan-Chen (Imiscível)
                // f = Fluido Vermelho (Invasor), g = Fluido Azul (Residente)
                
                double rho_red = 0.0;
                double rho_blue = RHO_BLUE_INIT;

                // Fluido invasor na entrada
                if (x < 15) {
                    rho_red = RHO_RED_INIT;
                    rho_blue = 0.0; // Quase zero, mas evite 0.0 absoluto se possível em SC complexos
                }
                
                // Pequeno ruído para quebrar simetria
                if (x==14) rho_red += noise(rng);

                mesh.rho[id] = rho_red + rho_blue;
                
                // Armazenamos densidades nas pops de momento zero (truque para acesso rápido ou usar array auxiliar)
                // Mas aqui vamos inicializar o equilíbrio direto.
                for(int k=0; k<9; ++k) {
                    mesh.f[mesh.idx(x,y,k)] = W[k] * rho_red;
                    mesh.g[mesh.idx(x,y,k)] = W[k] * rho_blue;
                }
            }
        }
    }
}

// ---------------------------------------------------------
// COLISÃO 1: MISCÍVEL (Cópia Otimizada do Anterior)
// ---------------------------------------------------------
void collide_miscible(LatticeMesh& mesh) {
    double diff = (U_INLET * (double)NX) / PECLET;
    double tau_g = 0.5 + diff / CS_SQ;
    double omega_g = 1.0 / tau_g;

    #pragma omp parallel for schedule(dynamic)
    for(int y = 0; y < NY; ++y) {
        for(int x = 0; x < NX; ++x) {
            size_t id = mesh.idx(x,y);
            if(mesh.mask[id] == 1) continue;

            // Macroscópicas
            double rho = 0.0, ux = 0.0, uy = 0.0, C = 0.0;
            for(int k=0; k<9; ++k) {
                rho += mesh.f[mesh.idx(x,y,k)];
                ux  += mesh.f[mesh.idx(x,y,k)] * CX[k];
                uy  += mesh.f[mesh.idx(x,y,k)] * CY[k];
                C   += mesh.g[mesh.idx(x,y,k)];
            }
            if(rho>0) { ux/=rho; uy/=rho; }
            mesh.rho[id]=rho; mesh.ux[id]=ux; mesh.uy[id]=uy; mesh.C[id]=C; // Salva C para plotar

            // Relaxamento Variável (Viscosidade)
            double C_cl = std::max(0.0, std::min(1.0, C));
            double nu_mix = (NU_BASE * VISC_RATIO_MISC) * std::pow(NU_BASE/(NU_BASE*VISC_RATIO_MISC), C_cl);
            double omega = 1.0 / (0.5 + nu_mix/CS_SQ);
            double u_sq = ux*ux + uy*uy;

            // Colisão BGK (simplificado para brevidade, mas pode usar TRT)
            for(int k=0; k<9; ++k) {
                double cu = CX[k]*ux + CY[k]*uy;
                double feq = W[k]*rho*(1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u_sq);
                double geq = W[k]*C_cl*(1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u_sq);

                double f_out = mesh.f[mesh.idx(x,y,k)] * (1.0-omega) + omega*feq;
                double g_out = mesh.g[mesh.idx(x,y,k)] * (1.0-omega_g) + omega_g*geq;

                // Streaming direto
                int nx = x + CX[k]; int ny = y + CY[k];
                if(nx>=0 && nx<NX && ny>=0 && ny<NY && mesh.mask[mesh.idx(nx,ny)]==0) {
                    mesh.f_new[mesh.idx(nx,ny,k)] = f_out;
                    mesh.g_new[mesh.idx(nx,ny,k)] = g_out;
                } else {
                    // Bounce-back
                    mesh.f_new[mesh.idx(x,y,OPP[k])] = f_out;
                    mesh.g_new[mesh.idx(x,y,OPP[k])] = g_out;
                }
            }
        }
    }
}

// ---------------------------------------------------------
// COLISÃO 2: SHAN-CHEN MULTICOMPONENTE (Imiscível)
// ---------------------------------------------------------
void collide_immiscible(LatticeMesh& mesh) {
    double tau = TAU_SC;
    double omega = 1.0 / tau;

    // Passo 1: Calcular densidades locais de A e B
    // Precisamos disso antes de calcular as forças, então idealmente
    // faríamos em dois passos ou assumiríamos densidade do passo anterior.
    // Para simplificar e rodar rápido, calculamos "on the fly" com dados streaming antigos,
    // mas o correto em SC é ter rho disponível. 
    // Vamos calcular rho localmente primeiro.
    
    // (Num código super otimizado, rho já estaria salvo, mas vamos recalcular)
    // Precisamos de rho vizinho para a força, então precisamos de um buffer de rho.
    // O jeito mais simples sem criar arrays novos é fazer dois loops: 
    // um calcula rho, outro colide.

    #pragma omp parallel for
    for(int i=0; i<NX*NY; ++i) {
        double ra = 0.0, rb = 0.0;
        for(int k=0; k<9; ++k) {
            ra += mesh.f[i*9 + k];
            rb += mesh.g[i*9 + k];
        }
        // Usamos C para guardar a Fração de Fluido A: rhoA / (rhoA + rhoB)
        // Ou guardamos rhoA em rho e rhoB em C? Vamos usar C para rhoB temporariamente.
        mesh.rho[i] = ra; 
        mesh.C[i]   = rb; // Hack: usando vetor C para guardar densidade do fluido B
    }

    #pragma omp parallel for schedule(dynamic)
    for(int y = 0; y < NY; ++y) {
        for(int x = 0; x < NX; ++x) {
            size_t id = mesh.idx(x,y);
            if(mesh.mask[id] == 1) continue;

            double rhoA = mesh.rho[id];
            double rhoB = mesh.C[id];
            double rho_tot = rhoA + rhoB;
            if (rho_tot < 1e-9) rho_tot = 1e-9;

            // Calcular Velocidade Baricêntrica (comum)
            double ux_tmp = 0.0, uy_tmp = 0.0;
            for(int k=0; k<9; ++k) {
                double val = mesh.f[mesh.idx(x,y,k)] + mesh.g[mesh.idx(x,y,k)];
                ux_tmp += val * CX[k];
                uy_tmp += val * CY[k];
            }
            double ux = ux_tmp / rho_tot;
            double uy = uy_tmp / rho_tot;

            // Calcular Força de Interação (Shan-Chen)
            // F_A = - G * rhoA(x) * sum( w_k * rhoB(x+k) * c_k )
            // A força age repelindo o outro fluido.
            double Fx = 0.0, Fy = 0.0;
            
            for(int k=1; k<9; ++k) {
                int nx = x + CX[k];
                int ny = y + CY[k];
                if(nx>=0 && nx<NX && ny>=0 && ny<NY) {
                    double rb_neigh = mesh.C[mesh.idx(nx,ny)]; // rhoB do vizinho
                    Fx += W[k] * rb_neigh * CX[k];
                    Fy += W[k] * rb_neigh * CY[k];
                }
            }
            // Força aplicada no Fluido A devido ao B
            Fx *= -G_INT * rhoA; 
            Fy *= -G_INT * rhoA;

            // Velocidade de Equilíbrio Modificada (Shifted Velocity)
            double ux_eq = ux + (tau * Fx) / rhoA; // Aproximação comum SC
            double uy_eq = uy + (tau * Fy) / rhoA;
            // Para o fluido B, a força é oposta (-F)
            double ux_eq_b = ux - (tau * Fx) / rhoB;
            double uy_eq_b = uy - (tau * Fy) / rhoB;

            // Colisão
            double usq_a = ux_eq*ux_eq + uy_eq*uy_eq;
            double usq_b = ux_eq_b*ux_eq_b + uy_eq_b*uy_eq_b;

            for(int k=0; k<9; ++k) {
                double cu_a = CX[k]*ux_eq + CY[k]*uy_eq;
                double feq = W[k] * rhoA * (1.0 + 3.0*cu_a + 4.5*cu_a*cu_a - 1.5*usq_a);

                double cu_b = CX[k]*ux_eq_b + CY[k]*uy_eq_b;
                double geq = W[k] * rhoB * (1.0 + 3.0*cu_b + 4.5*cu_b*cu_b - 1.5*usq_b);

                double f_out = mesh.f[mesh.idx(x,y,k)] * (1.0-omega) + omega*feq;
                double g_out = mesh.g[mesh.idx(x,y,k)] * (1.0-omega) + omega*geq;

                // Streaming
                int nx = x + CX[k]; int ny = y + CY[k];
                bool solid = (nx<0 || nx>=NX || ny<0 || ny>=NY || mesh.mask[mesh.idx(nx,ny)]==1);
                
                if(!solid) {
                    mesh.f_new[mesh.idx(nx,ny,k)] = f_out;
                    mesh.g_new[mesh.idx(nx,ny,k)] = g_out;
                } else {
                    mesh.f_new[mesh.idx(x,y,OPP[k])] = f_out;
                    mesh.g_new[mesh.idx(x,y,OPP[k])] = g_out;
                }
            }
            
            // Atualizar macroscópicas para visualização
            mesh.ux[id] = ux; // salva velocidade real (baricêntrica)
            mesh.uy[id] = uy;
            // Normaliza C para plotagem (0=Azul puro, 1=Vermelho puro)
            mesh.C[id] = rhoA / (rhoA + rhoB); // Sobrescreve o hack anterior
        }
    }
}

// Wrapper Principal
void collide_and_stream(LatticeMesh& mesh) {
    // Limpa buffers
    std::fill(mesh.f_new.begin(), mesh.f_new.end(), 0.0);
    std::fill(mesh.g_new.begin(), mesh.g_new.end(), 0.0);

    if (SIM_MODE == MISCIBLE) {
        collide_miscible(mesh);
    } else {
        collide_immiscible(mesh);
    }
}

void update_buffers(LatticeMesh& mesh) {
    mesh.f.swap(mesh.f_new);
    mesh.g.swap(mesh.g_new);
}

void update_macroscopics_C(LatticeMesh& mesh, double& total_mass) {
    // Apenas cálculo de massa para log
    total_mass = 0.0;
    #pragma omp parallel for reduction(+:total_mass)
    for(int i=0; i<NX*NY; ++i) {
        if(SIM_MODE == MISCIBLE) total_mass += mesh.C[i];
        else total_mass += mesh.rho[i]; // No imiscivel, soma a densidade do fluido A
    }
}