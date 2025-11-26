// main_immiscible_zouhe_outlet_pressure.cpp
// Versão com Zou-He completo no inlet e outlet por pressão (rho fixo).
// Mantém pseudopotential + Guo forcing + limpeza de f_new + clamps.

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <omp.h>
#include <algorithm>
#include <sys/stat.h>
#include <cstdlib>
#include <cstring>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ============================================================================
// PARÂMETROS
// ============================================================================
constexpr int NX = 400;
constexpr int NY = 150;
constexpr int MAX_ITER = 30000;
constexpr int OUTPUT_FREQ = 200;

constexpr double G_INT = 1.0;          // Força Shan-Chen
constexpr double RHO_RED_INIT = 1.0;
constexpr double RHO_BLUE_INIT = 1.0;
constexpr double BACKGROUND_RHO = 0.05; // fundo mínimo
constexpr double TAU = 1.0;
constexpr double U_INLET = 0.04;
constexpr double POROSITY = 0.75;
constexpr int SEED = 42;

// Outlet pressure (densidade total fixa)
constexpr double RHO_OUTLET = 1.0; // densidade total prescrita na saída

// D2Q9
constexpr double CS_SQ = 1.0/3.0;
const double W[9] = {4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.};
const int CX[9]   = {0, 1, 0, -1, 0, 1, -1, -1, 1};
const int CY[9]   = {0, 0, 1,  0,-1, 1,  1, -1,-1};
const int OPP[9]  = {0, 3, 4, 1, 2, 7, 8, 5, 6};

// ============================================================================
// ESTRUTURA
// ============================================================================
struct LatticeMesh {
    std::vector<double> fA, fA_new;
    std::vector<double> fB, fB_new;
    std::vector<double> rhoA, rhoB;
    std::vector<double> ux, uy;
    std::vector<int> mask;

    LatticeMesh() {
        size_t N = NX * NY;
        fA.resize(N * 9, 0.0); fA_new.resize(N * 9, 0.0);
        fB.resize(N * 9, 0.0); fB_new.resize(N * 9, 0.0);
        rhoA.resize(N, 0.0); rhoB.resize(N, 0.0);
        ux.resize(N, 0.0); uy.resize(N, 0.0);
        mask.resize(N, 0);
    }

    // Layout NODE-MAJOR: ((y*NX + x) * 9 + k)
    inline size_t idx(int x, int y, int k=0) const {
        return ( (size_t)y * NX + (size_t)x ) * 9 + (size_t)k;
    }

    inline size_t nid(int x, int y) const {
        return (size_t)y * NX + (size_t)x;
    }
};

// ============================================================================
// UTIL
// ============================================================================
static inline uint8_t clamp8(double v) {
    if (v < 0.0) return 0; if (v > 255.0) return 255;
    return static_cast<uint8_t>(v + 0.5);
}

void create_dirs() {
    #ifdef _WIN32
        system("mkdir output 2> NUL");
        system("mkdir output\\imagens 2> NUL");
    #else
        mkdir("output", 0777);
        mkdir("output/imagens", 0777);
    #endif
}

void save_frame(const LatticeMesh& mesh, int step) {
    std::vector<uint8_t> img(NX * NY * 3);
    std::string fname = "output/imagens/sim_" + std::to_string(step) + ".png";

    #pragma omp parallel for
    for(int y=0; y<NY; ++y) {
        for(int x=0; x<NX; ++x) {
            size_t id = y*NX + x;
            size_t pid = id * 3;

            if (mesh.mask[id] == 1) { // solid
                img[pid]=0; img[pid+1]=0; img[pid+2]=0;
                continue;
            }

            double rA = mesh.rhoA[id];
            double rB = mesh.rhoB[id];
            double sum = rA + rB;
            if(sum < 1e-9) sum = 1.0;
            double fraction = rA / sum;

            uint8_t r = clamp8(fraction * 255.0);
            uint8_t b = clamp8((1.0 - fraction) * 255.0);
            uint8_t g = 50;

            if (std::abs(fraction - 0.5) < 0.05) {
                r = 0; g = 0; b = 0;
            }

            img[pid] = r; img[pid+1] = g; img[pid+2] = b;
        }
    }
    stbi_write_png(fname.c_str(), NX, NY, 3, img.data(), NX * 3);
}

// ============================================================================
// SIMULAÇÃO
// ============================================================================
void initialize(LatticeMesh& mesh) {
    srand(SEED);
    for(int i=0; i<NX*NY; ++i) {
        double r = (double)rand()/RAND_MAX;
        mesh.mask[i] = (r > POROSITY) ? 1 : 0;
    }
    // limpa entrada/saida (garante caminho livre)
    for(int y=0; y<NY; ++y) {
        for(int x=0; x<NX; ++x) {
            if(x<15 || x>NX-15) mesh.mask[y*NX + x] = 0;
        }
    }

    for(int y=0; y<NY; ++y) {
        for(int x=0; x<NX; ++x) {
            size_t nid = mesh.nid(x,y);
            double rhoA = BACKGROUND_RHO;
            double rhoB = RHO_BLUE_INIT;
            if (x < 20) {
                rhoA = RHO_RED_INIT;
                rhoB = BACKGROUND_RHO;
                if (x == 19) rhoA += ((double)rand()/RAND_MAX - 0.5)*0.1;
            }
            mesh.rhoA[nid] = rhoA;
            mesh.rhoB[nid] = rhoB;
            mesh.ux[nid] = 0.0;
            mesh.uy[nid] = 0.0;

            for(int k=0; k<9; ++k) {
                mesh.fA[mesh.idx(x,y,k)] = W[k] * rhoA;
                mesh.fB[mesh.idx(x,y,k)] = W[k] * rhoB;
            }
        }
    }
}

// psuedopotential psi(ρ)
inline double psi_of_rho(double rho) {
    return 1.0 - std::exp(-rho);
}

void collide_and_stream(LatticeMesh& mesh) {
    double omega = 1.0 / TAU;
    size_t N = NX * NY;

    // Zerar f_new (muito importante)
    #pragma omp parallel for
    for(size_t i=0; i < N*9; ++i) {
        mesh.fA_new[i] = 0.0;
        mesh.fB_new[i] = 0.0;
    }

    // --- PASSO 1: macroscópicas ---
    #pragma omp parallel for
    for(int i=0; i<NX*NY; ++i) {
        double ra = 0.0, rb = 0.0;
        for(int k=0; k<9; ++k) {
            ra += mesh.fA[i*9 + k];
            rb += mesh.fB[i*9 + k];
        }
        // clamp inicial para evitar negativos
        if (ra < 0.0) ra = 0.0;
        if (rb < 0.0) rb = 0.0;
        mesh.rhoA[i] = ra;
        mesh.rhoB[i] = rb;
    }

    // --- PASSO 2: colisão + forçamento (Guo) + streaming ---
    #pragma omp parallel for schedule(dynamic)
    for(int y=0; y<NY; ++y) {
        for(int x=0; x<NX; ++x) {
            size_t id = mesh.nid(x,y);
            if (mesh.mask[id] == 1) continue;

            double rhoA = mesh.rhoA[id];
            double rhoB = mesh.rhoB[id];

            // limite inferior para densidade (evita vácuos numéricos)
            const double min_rho = 1e-8;
            if (rhoA < min_rho) rhoA = min_rho;
            if (rhoB < min_rho) rhoB = min_rho;
            mesh.rhoA[id] = rhoA;
            mesh.rhoB[id] = rhoB;

            // momento total
            double momX = 0.0, momY = 0.0;
            for(int k=0; k<9; ++k) {
                double val = mesh.fA[id*9 + k] + mesh.fB[id*9 + k];
                momX += val * CX[k];
                momY += val * CY[k];
            }

            double rho_tot = rhoA + rhoB;
            if (rho_tot < 1e-12) rho_tot = 1e-12;
            double ux = momX / rho_tot;
            double uy = momY / rho_tot;

            // --- Shan-Chen force (usando psi) ---
            double psiA = psi_of_rho(rhoA);
            double psiB = psi_of_rho(rhoB);

            double ForceA_x = 0.0, ForceA_y = 0.0;
            // soma sobre vizinhos (k=1..8)
            for(int k=1; k<9; ++k) {
                int nx = x + CX[k];
                int ny = y + CY[k];
                if (nx>=0 && nx<NX && ny>=0 && ny<NY) {
                    size_t nidn = mesh.nid(nx,ny);
                    double rhoB_neigh = mesh.rhoB[nidn];
                    double psiB_neigh = psi_of_rho(rhoB_neigh);
                    ForceA_x += W[k] * psiB_neigh * CX[k];
                    ForceA_y += W[k] * psiB_neigh * CY[k];
                }
            }
            // multiplicador Shan-Chen (A sente B)
            ForceA_x *= -G_INT * psiA;
            ForceA_y *= -G_INT * psiA;
            // Força no B é oposta (par conservativo)
            double ForceB_x = -ForceA_x;
            double ForceB_y = -ForceA_y;

            // --- Colisão BGK + Guo forcing para cada componente ---
            double u_dot_F_A = ux * ForceA_x + uy * ForceA_y;
            double u_dot_F_B = ux * ForceB_x + uy * ForceB_y;

            for(int k=0; k<9; ++k) {
                // Equilíbrio com velocidade baricêntrica (usar ux,uy)
                double cu = CX[k]*ux + CY[k]*uy;
                // Para A:
                double feqA = W[k] * rhoA * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*(ux*ux + uy*uy));
                // Para B:
                double feqB = W[k] * rhoB * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*(ux*ux + uy*uy));

                // Termo de forçamento tipo Guo
                double eF_A = CX[k]*ForceA_x + CY[k]*ForceA_y;
                double eF_B = CX[k]*ForceB_x + CY[k]*ForceB_y;

                double Su_A = (1.0 - 0.5*omega) * W[k] * ( 3.0*eF_A + 9.0*cu*eF_A - 3.0*u_dot_F_A );
                double Su_B = (1.0 - 0.5*omega) * W[k] * ( 3.0*eF_B + 9.0*cu*eF_B - 3.0*u_dot_F_B );

                double fA_old = mesh.fA[id*9 + k];
                double fB_old = mesh.fB[id*9 + k];

                double fA_post = fA_old * (1.0 - omega) + omega * feqA + Su_A;
                double fB_post = fB_old * (1.0 - omega) + omega * feqB + Su_B;

                // Streaming
                int nx = x + CX[k];
                int ny = y + CY[k];
                bool is_solid = false;
                if (nx<0 || nx>=NX || ny<0 || ny>=NY) is_solid = true;
                else if (mesh.mask[mesh.nid(nx,ny)] == 1) is_solid = true;

                if (is_solid) {
                    // bounce-back: devolve para direção oposta no mesmo nó
                    mesh.fA_new[ mesh.idx(x,y,OPP[k]) ] = fA_post;
                    mesh.fB_new[ mesh.idx(x,y,OPP[k]) ] = fB_post;
                } else {
                    mesh.fA_new[ mesh.idx(nx,ny,k) ] = fA_post;
                    mesh.fB_new[ mesh.idx(nx,ny,k) ] = fB_post;
                }
            } // fim k
        } // fim x
    } // fim y
}

// ======================
// BOUNDARIES (Zou-He inlet + Pressure outlet)
// ======================
void apply_boundary(LatticeMesh& mesh) {
    // --- INLET (x = 0) : Zou-He completa (velocidade prescrita) ---
    // Unknown directions: k=1,5,8 (cx = +1)
    double ux_in = U_INLET;
    double uy_in = 0.0;
    double rhoA_in = 1.0;
    double rhoB_in = BACKGROUND_RHO;

    #pragma omp parallel for
    for(int y=1; y<NY-1; ++y) {
        int x = 0;
        size_t nid = mesh.nid(x,y);
        if (mesh.mask[nid] == 1) continue;

        // Para cada componente, aplique Zou-He usando rho_comp_in e u_in.
        // COMPONENTE A
        {
            // Conhecidos: f0,f2,f4,f3,f6,f7 em fA_new (direções com cx<=0)
            double f0 = mesh.fA_new[mesh.idx(x,y,0)];
            double f2 = mesh.fA_new[mesh.idx(x,y,2)];
            double f4 = mesh.fA_new[mesh.idx(x,y,4)];
            double f3 = mesh.fA_new[mesh.idx(x,y,3)];
            double f6 = mesh.fA_new[mesh.idx(x,y,6)];
            double f7 = mesh.fA_new[mesh.idx(x,y,7)];

            // Defina rhoA no inlet conforme desejado
            double rhoA = rhoA_in;
            // calcula as populações desconhecidas:
            // f1 = f3 + 2/3 * rho * ux
            mesh.fA_new[mesh.idx(x,y,1)] = f3 + (2.0/3.0) * rhoA * ux_in;
            // f5 = f7 + 0.5*(f2 - f4) + (1/6)*rho*(ux+uy)
            mesh.fA_new[mesh.idx(x,y,5)] = f7 + 0.5*(f2 - f4) + (1.0/6.0) * rhoA * (ux_in + uy_in);
            // f8 = f6 + 0.5*(f4 - f2) + (1/6)*rho*(ux-uy)
            mesh.fA_new[mesh.idx(x,y,8)] = f6 + 0.5*(f4 - f2) + (1.0/6.0) * rhoA * (ux_in - uy_in);

            // Atualiza macroscópicas locais (opcional aqui)
            mesh.rhoA[nid] = rhoA;
        }

        // COMPONENTE B
        {
            double f0 = mesh.fB_new[mesh.idx(x,y,0)];
            double f2 = mesh.fB_new[mesh.idx(x,y,2)];
            double f4 = mesh.fB_new[mesh.idx(x,y,4)];
            double f3 = mesh.fB_new[mesh.idx(x,y,3)];
            double f6 = mesh.fB_new[mesh.idx(x,y,6)];
            double f7 = mesh.fB_new[mesh.idx(x,y,7)];

            double rhoB = rhoB_in;
            mesh.fB_new[mesh.idx(x,y,1)] = f3 + (2.0/3.0) * rhoB * ux_in;
            mesh.fB_new[mesh.idx(x,y,5)] = f7 + 0.5*(f2 - f4) + (1.0/6.0) * rhoB * (ux_in + uy_in);
            mesh.fB_new[mesh.idx(x,y,8)] = f6 + 0.5*(f4 - f2) + (1.0/6.0) * rhoB * (ux_in - uy_in);

            mesh.rhoB[nid] = rhoB;
        }

        // Opcional: podemos ajustar velocidades macroscópicas do nó
        double rho_tot = mesh.rhoA[nid] + mesh.rhoB[nid];
        if (rho_tot < 1e-12) rho_tot = 1e-12;
        mesh.ux[nid] = ux_in;
        mesh.uy[nid] = uy_in;
    }

    // --- OUTLET (x = NX-1) : condição de pressão (rho_total fixo)
    // Unknown directions: k=3,6,7 (cx = -1)
    // Estratégia: fixar rho_total = RHO_OUTLET; obter frações via extrapolação do vizinho x=NX-2
    #pragma omp parallel for
    for(int y=1; y<NY-1; ++y) {
        int x = NX-1;
        size_t nid = mesh.nid(x,y);
        if (mesh.mask[nid] == 1) continue;

        // Vizinho interior
        int x_in = NX-2;
        size_t nid_in = mesh.nid(x_in,y);
        // Frações baseadas no vizinho (evita forçar composição arbitrária)
        double rhoA_neigh = mesh.rhoA[nid_in];
        double rhoB_neigh = mesh.rhoB[nid_in];
        double rho_neigh_tot = rhoA_neigh + rhoB_neigh;
        double fracA = 0.0;
        if (rho_neigh_tot > 1e-12) fracA = rhoA_neigh / rho_neigh_tot;
        double rho_tot = RHO_OUTLET;
        double rhoA_out = std::max(1e-12, fracA * rho_tot);
        double rhoB_out = std::max(1e-12, rho_tot - rhoA_out);

        // Primeiro: precisamos conhecer as populações "conhecidas" no outlet (direções com cx>=0):
        // para cada componente, use f_new já presente: f0,f1,f2,f4,f5,f8 (cx>=0)
        // Calcule u_x a partir das equações de Zou-He (pressão):
        // u_x = -1 + (1/rho) * (f0 + f2 + f4 + 2*(f1 + f5 + f8))
        // Aplicamos para a mistura para obter u_x,u_y (usamos somas das componentes)
        double f0_sum = mesh.fA_new[mesh.idx(x,y,0)] + mesh.fB_new[mesh.idx(x,y,0)];
        double f1_sum = mesh.fA_new[mesh.idx(x,y,1)] + mesh.fB_new[mesh.idx(x,y,1)];
        double f2_sum = mesh.fA_new[mesh.idx(x,y,2)] + mesh.fB_new[mesh.idx(x,y,2)];
        double f4_sum = mesh.fA_new[mesh.idx(x,y,4)] + mesh.fB_new[mesh.idx(x,y,4)];
        double f5_sum = mesh.fA_new[mesh.idx(x,y,5)] + mesh.fB_new[mesh.idx(x,y,5)];
        double f8_sum = mesh.fA_new[mesh.idx(x,y,8)] + mesh.fB_new[mesh.idx(x,y,8)];

        double ux_out = -1.0 + (1.0 / rho_tot) * ( f0_sum + f2_sum + f4_sum + 2.0*(f1_sum + f5_sum + f8_sum) );

        // podemos estimar uy approximando pelo vizinho
        double uy_out = mesh.uy[nid_in];

        // Agora reconstruir as populações desconhecidas por componente (k = 3,6,7)
        // Fórmulas (espelho da Zou-He):
        // f3 = f1 - 2/3 * rho * u_x
        // f6 = f8 + 0.5*(f4 - f2) - (1/6) * rho * (u_x + u_y)
        // f7 = f5 + 0.5*(f2 - f4) - (1/6) * rho * (u_x - u_y)

        // COMPONENTE A
        {
            double f1 = mesh.fA_new[mesh.idx(x,y,1)];
            double f2 = mesh.fA_new[mesh.idx(x,y,2)];
            double f4 = mesh.fA_new[mesh.idx(x,y,4)];
            double f5 = mesh.fA_new[mesh.idx(x,y,5)];
            double f8 = mesh.fA_new[mesh.idx(x,y,8)];

            double rhoA = rhoA_out;

            mesh.fA_new[mesh.idx(x,y,3)] = f1 - (2.0/3.0) * rhoA * ux_out;
            mesh.fA_new[mesh.idx(x,y,6)] = f8 + 0.5*(f4 - f2) - (1.0/6.0) * rhoA * (ux_out + uy_out);
            mesh.fA_new[mesh.idx(x,y,7)] = f5 + 0.5*(f2 - f4) - (1.0/6.0) * rhoA * (ux_out - uy_out);

            mesh.rhoA[nid] = rhoA;
        }

        // COMPONENTE B
        {
            double f1 = mesh.fB_new[mesh.idx(x,y,1)];
            double f2 = mesh.fB_new[mesh.idx(x,y,2)];
            double f4 = mesh.fB_new[mesh.idx(x,y,4)];
            double f5 = mesh.fB_new[mesh.idx(x,y,5)];
            double f8 = mesh.fB_new[mesh.idx(x,y,8)];

            double rhoB = rhoB_out;

            mesh.fB_new[mesh.idx(x,y,3)] = f1 - (2.0/3.0) * rhoB * ux_out;
            mesh.fB_new[mesh.idx(x,y,6)] = f8 + 0.5*(f4 - f2) - (1.0/6.0) * rhoB * (ux_out + uy_out);
            mesh.fB_new[mesh.idx(x,y,7)] = f5 + 0.5*(f2 - f4) - (1.0/6.0) * rhoB * (ux_out - uy_out);

            mesh.rhoB[nid] = rhoB;
        }

        mesh.ux[nid] = ux_out;
        mesh.uy[nid] = uy_out;
    }

    // Para evitar boder effects, também preenchemos as bordas y=0 e y=NY-1 com bounce-back simples
    // (alternativa: parelhos sólidos já cobrem isso se mask estiver definido).
    // Opcional: já tratado nos passos de colisão/streaming por bounce-back via mask.
}

// atualiza ponteiros (swap)
void update_pointers(LatticeMesh& mesh) {
    mesh.fA.swap(mesh.fA_new);
    mesh.fB.swap(mesh.fB_new);
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    create_dirs();
    std::cout << "=== LBM Shan-Chen Imiscivel (Zou-He inlet + Pressure outlet) ===" << std::endl;
    std::cout << "Dimensoes: " << NX << "x" << NY << std::endl;
    std::cout << "G (Interacao): " << G_INT << std::endl;

    LatticeMesh mesh;
    initialize(mesh);

    for(int step=0; step<=MAX_ITER; ++step) {
        collide_and_stream(mesh);
        apply_boundary(mesh);
        update_pointers(mesh);

        if(step % OUTPUT_FREQ == 0) {
            // Calcula massa total A e estatísticas de densidade para debug
            double totalMass = 0.0;
            double minA = 1e30, maxA = -1e30;
            double minB = 1e30, maxB = -1e30;
            for(size_t i=0; i<mesh.rhoA.size(); ++i) {
                totalMass += mesh.rhoA[i];
                minA = std::min(minA, mesh.rhoA[i]);
                maxA = std::max(maxA, mesh.rhoA[i]);
                minB = std::min(minB, mesh.rhoB[i]);
                maxB = std::max(maxB, mesh.rhoB[i]);
            }
            std::cout << "Step: " << step << " | Massa A: " << totalMass
                      << " | rhoA(min,max): " << minA << "," << maxA
                      << " | rhoB(min,max): " << minB << "," << maxB << std::endl;
            save_frame(mesh, step);
        }
    }

    std::cout << "Fim da simulacao." << std::endl;
    return 0;
}
