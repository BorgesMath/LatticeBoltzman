// main.cpp
// Versão corrigida: gera PNG direto (concentração + velocidade) usando stb_image_write.
// Baixe stb_image_write.h na mesma pasta:
// curl -L -o stb_image_write.h https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
// Compile: g++ -O3 -fopenmp main.cpp -o lbm_sim

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <random>
#include <omp.h>
#include <algorithm>
#include <iomanip>
#include <queue>
#include <cstdint>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// PARÂMETROS DA SIMULAÇÃO (Configuração Física e Numérica)
// ============================================================================
constexpr int NX = 400;             // Largura do domínio
constexpr int NY = 150;             // Altura do domínio
constexpr int MAX_ITER = 30000;     // Total de passos de tempo
constexpr int OUTPUT_FREQ = 200;    // Frequência de escrita (imagens)

// Parâmetros Físicos (pode ajustar conforme necessidade)
constexpr double RE_COARSE = 0.2;   // Número de Reynolds (grosseiro)
constexpr double VISC_RATIO = 50.0; // Razão de viscosidade (Óleo / CO2)
constexpr double PECLET = 100.0;    // Número de Peclet (Advecção / Difusão)

// Propriedades do CO2 (Fluido Invasor - Baixa Viscosidade)
constexpr double NU_CO2 = 0.005;    // Viscosidade cinemática CO2 (Lattice Units)
constexpr double RHO0 = 1.0;        // Densidade base

// Parâmetros LBM
constexpr double CS_SQ = 1.0 / 3.0; // Velocidade do som ao quadrado
constexpr double MAGIC_LAMBDA = 0.25;// Parâmetro mágico para estabilidade TRT
constexpr double U_INLET = 0.02;    // Velocidade de injeção (Lattice Units) - reduzido para estabilidade

// Parâmetros do Meio Poroso
constexpr double POROSITY_TARGET = 0.75; // Porosidade alvo
constexpr int OBSTACLE_SEED = 42;        // Semente aleatória

// ============================================================================
// CONSTANTES DO LATTICE D2Q9
// ============================================================================
const double W[9] = {
    4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.
};

const int CX[9] = {0, 1, 0, -1, 0,  1, -1, -1,  1};
const int CY[9] = {0, 0, 1,  0, -1, 1,  1, -1, -1};
const int OPP[9] = {0, 3, 4,  1,  2, 7,  8,  5,  6}; // Direção oposta

// ============================================================================
// ESTRUTURAS DE DADOS (SoA - Structure of Arrays para Cache Efficiency)
// ============================================================================
struct LatticeMesh {
    // Populações Hidrodinâmicas (f) - 2 Buffers para Streaming
    std::vector<double> f;
    std::vector<double> f_new;

    // Populações de Concentração (g) - 2 Buffers
    std::vector<double> g;
    std::vector<double> g_new;

    // Campos Macroscópicos
    std::vector<double> rho;    // Densidade
    std::vector<double> ux;     // Velocidade X
    std::vector<double> uy;     // Velocidade Y
    std::vector<double> C;      // Concentração (0.0 = Óleo, 1.0 = CO2)
    std::vector<double> nu_loc; // Viscosidade local (varia com C)

    // Geometria (0 = Fluido, 1 = Sólido)
    std::vector<int> mask;

    LatticeMesh() {
        size_t N = NX * NY;
        f.resize(N * 9, 0.0);
        f_new.resize(N * 9, 0.0);
        g.resize(N * 9, 0.0);
        g_new.resize(N * 9, 0.0);
        rho.resize(N, RHO0);
        ux.resize(N, 0.0);
        uy.resize(N, 0.0);
        C.resize(N, 0.0);
        nu_loc.resize(N, NU_CO2 * VISC_RATIO); // Começa cheio de óleo
        mask.resize(N, 0);
    }

    // Helper para indexação 1D
    inline size_t idx(int x, int y, int k = 0) const {
        // Para acessos macroscópicos (k==0), retorna y*NX + x
        // Para f/g (k in [0..8]), retorna k*(NX*NY) + y*NX + x
        return static_cast<size_t>(k) * (NX * NY) + static_cast<size_t>(y) * NX + static_cast<size_t>(x);
    }
};

// ============================================================================
// FUNÇÕES AUXILIARES
// ============================================================================
void generate_porous_media(LatticeMesh& mesh) {
    std::mt19937 rng(OBSTACLE_SEED);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // 1. Ruído Aleatório
    for(int y = 0; y < NY; ++y) {
        for(int x = 0; x < NX; ++x) {
            // Deixa entrada e saída livres
            if (x < 10 || x > NX - 10) {
                mesh.mask[mesh.idx(x, y)] = 0;
                continue;
            }

            if (dist(rng) > POROSITY_TARGET) {
                mesh.mask[mesh.idx(x, y)] = 1;
            } else {
                mesh.mask[mesh.idx(x, y)] = 0;
            }
        }
    }

    // 2. Autômato Celular para agrupar obstáculos (criar poros mais realistas)
    std::vector<int> temp_mask = mesh.mask;
    for (int iter = 0; iter < 4; ++iter) { // 4 passadas de suavização
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
                if (mesh.mask[id] == 1) {
                    temp_mask[id] = (neighbors >= 4)? 1 : 0;
                } else {
                    temp_mask[id] = (neighbors >= 5)? 1 : 0;
                }
            }
        }
        mesh.mask = temp_mask;
    }
    std::cout << "-> Meio poroso gerado." << std::endl;
}

// Inicialização das populações
void initialize(LatticeMesh& mesh) {
    double nu_oil = NU_CO2 * VISC_RATIO;

    // Inicial condição: domínio cheio de óleo (C=0), entrada como CO2 (C~1 com pequena perturbação)
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> rdist(-0.02, 0.02); // pequena perturbação

    for(int y = 0; y < NY; ++y) {
        for(int x = 0; x < NX; ++x) {
            size_t id = mesh.idx(x, y);

            // Inicializa campos macroscópicos
            mesh.rho[id] = RHO0;
            mesh.ux[id] = 0.0;
            mesh.uy[id] = 0.0;
            mesh.C[id] = 0.0; // Domínio cheio de óleo
            mesh.nu_loc[id] = nu_oil;

            // Perturbação na entrada para quebrar simetria inicial
            if (x == 0) {
                // pequena perturbação sinusoidal + ruído
                double perturb = 0.03 * std::sin(2.0*M_PI * y / (double)NY) + rdist(rng);
                double cval = 1.0 + perturb;
                if (cval < 0.0) cval = 0.0;
                if (cval > 1.0) cval = 1.0;
                mesh.C[id] = cval;
            }

            // Calcula Equilíbrio Inicial
            double u_sq = 0.0;
            for(int k=0; k<9; ++k) {
                double cu = 0.0; // ux=uy=0
                double term = 1.0 - 1.5*u_sq; // primeiro termo simplificado
                double feq = W[k] * RHO0 * term;
                double geq = W[k] * mesh.C[id] * term;

                mesh.f[mesh.idx(x,y,k)] = feq;
                mesh.g[mesh.idx(x,y,k)] = geq;
            }
        }
    }
}

// ============================================================================
// KERNELS LBM (Colisão e Streaming)
// ============================================================================
void collide_and_stream(LatticeMesh& mesh) {
    double nu_oil = NU_CO2 * VISC_RATIO;
    // Difusividade baseada no Peclet: D = (U * L) / Pe. Use L = NX para escala longitudinal.
    double diff = (U_INLET * (double)NX) / PECLET;
    double tau_g = 0.5 + diff / CS_SQ; // Relaxação para concentração (BGK)
    if (tau_g <= 0.5001) tau_g = 0.5001;
    double omega_g = 1.0 / tau_g;

    // Zera buffers de streaming para evitar escrita "fantasma"
    std::fill(mesh.f_new.begin(), mesh.f_new.end(), 0.0);
    std::fill(mesh.g_new.begin(), mesh.g_new.end(), 0.0);

    #pragma omp parallel for schedule(dynamic)
    for(int y = 0; y < NY; ++y) {
        for(int x = 0; x < NX; ++x) {
            size_t id_node = mesh.idx(x,y);

            // Se for sólido, aplicar bounce-back "estático": populações permanecem (ou zero)
            if(mesh.mask[id_node] == 1) {
                // Mantemos macroscópicas neutras; populações são tratadas no streaming por bounce-back
                continue;
            }

            // 1. Carregar Populações e Calcular Macroscópicas
            double rho = 0.0;
            double ux = 0.0, uy = 0.0;
            double C = 0.0;

            for(int k=0; k<9; ++k) {
                double val_f = mesh.f[mesh.idx(x,y,k)];
                double val_g = mesh.g[mesh.idx(x,y,k)];
                rho += val_f;
                ux  += val_f * CX[k];
                uy  += val_f * CY[k];
                C   += val_g;
            }

            // Proteção contra rho muito pequeno (evita divisão por zero)
            if (rho <= 1e-12) {
                // Reinicializa densidade e populações de equilíbrio (pequena correção)
                rho = RHO0;
                ux = 0.0;
                uy = 0.0;
            } else {
                ux /= rho;
                uy /= rho;
            }

            // Armazena macroscópicas para visualização e passo seguinte (C será normalizado abaixo)
            mesh.rho[id_node] = rho;
            mesh.ux[id_node]  = ux;
            mesh.uy[id_node]  = uy;

            // 2. Atualizar Viscosidade (Lei de Arrhenius / Mistura Logarítmica)
            double ratio = NU_CO2 / (NU_CO2 * VISC_RATIO);
            double C_phys = C;
            if (C_phys < 0.0) C_phys = 0.0;
            if (C_phys > 1.0) C_phys = 1.0;
            double nu_oil_local = NU_CO2 * VISC_RATIO;
            double nu_mix = nu_oil_local * std::pow(ratio, C_phys);
            mesh.nu_loc[id_node] = nu_mix;

            double tau_plus = 0.5 + nu_mix / CS_SQ;
            if (tau_plus <= 0.5001) tau_plus = 0.5001; // segurança
            double tau_minus = 0.5 + MAGIC_LAMBDA / (tau_plus - 0.5);
            if (!std::isfinite(tau_minus) || tau_minus <= 0.5001) tau_minus = tau_plus; // fallback
            double omega_plus = 1.0 / tau_plus;
            double omega_minus = 1.0 / tau_minus;

            double u_sq = ux*ux + uy*uy;

            // 3. Colisão e Streaming (Push)
            for(int k=0; k<9; ++k) {
                int k_opp = OPP[k];
                size_t idx_curr = mesh.idx(x,y,k);
                size_t idx_opp  = mesh.idx(x,y,k_opp);

                double fk = mesh.f[idx_curr];
                double fk_opp = mesh.f[idx_opp];

                double cu = CX[k]*ux + CY[k]*uy;

                // Equilíbrio (par e ímpar)
                double feq_sym = W[k] * rho * (1.0 + 4.5*cu*cu - 1.5*u_sq);
                double feq_asym = W[k] * rho * (3.0 * cu);
                double feq_k = feq_sym + feq_asym;

                // Decomposição TRT
                double f_sym = 0.5 * (fk + fk_opp);
                double f_asym = 0.5 * (fk - fk_opp);

                double f_post = fk
                              - omega_plus * (f_sym - feq_sym)
                              - omega_minus * (f_asym - feq_asym);

                // --- BGK para Concentração (g) ---
                double geq_k = W[k] * C_phys * (1.0 + 3.0*cu);
                double gk = mesh.g[idx_curr];
                double g_post = gk * (1.0 - omega_g) + omega_g * geq_k;

                // Streaming (push)
                int next_x = x + CX[k];
                int next_y = y + CY[k];

                // Caso destino esteja fora do domínio ou seja sólido -> bounce-back
                bool outside = (next_x < 0 || next_x >= NX || next_y < 0 || next_y >= NY);
                if (outside) {
                    // Escreve no nó atual na direção oposta (bounce-back)
                    mesh.f_new[mesh.idx(x, y, k_opp)] += f_post;
                    mesh.g_new[mesh.idx(x, y, k_opp)] += g_post;
                } else {
                    size_t id_next = mesh.idx(next_x, next_y);
                    if (mesh.mask[id_next] == 1) {
                        // destino é sólido -> bounce-back no nó atual
                        mesh.f_new[mesh.idx(x, y, k_opp)] += f_post;
                        mesh.g_new[mesh.idx(x, y, k_opp)] += g_post;
                    } else {
                        // destino é fluido -> streaming normal
                        mesh.f_new[mesh.idx(next_x, next_y, k)] += f_post;
                        mesh.g_new[mesh.idx(next_x, next_y, k)] += g_post;
                    }
                }
            } // fim do loop k
        }
    } // fim do loop x,y
}

// Condições de Contorno Específicas (Inlet/Outlet)
void apply_boundary_conditions(LatticeMesh& mesh) {
    #pragma omp parallel for
    for(int y = 0; y < NY; ++y) {
        if (mesh.mask[mesh.idx(0,y)] == 1) continue;

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

        double perturb = 0.02 * std::sin(2.0*M_PI * y / (double)NY);
        double C_in = 1.0 + perturb;
        if (C_in < 0.0) C_in = 0.0;
        if (C_in > 1.0) C_in = 1.0;

        for(int k = 0; k < 9; ++k) {
           double cu = CX[k]*U_INLET; // uy=0
           double geq = W[k] * C_in * (1.0 + 3.0*cu); // equilíbrio simplificado
           mesh.g_new[mesh.idx(0,y,k)] = geq;
        }

        mesh.C[mesh.idx(0,y)] = C_in;

        if (mesh.mask[mesh.idx(NX-1,y)] == 0) {
            for(int k=0; k<9; ++k) {
                mesh.f_new[mesh.idx(NX-1,y,k)] = mesh.f_new[mesh.idx(NX-2,y,k)];
                mesh.g_new[mesh.idx(NX-1,y,k)] = mesh.g_new[mesh.idx(NX-2,y,k)];
            }
        }
    }
}

// Atualização dos ponteiros (Ping-Pong)
void update_buffers(LatticeMesh& mesh) {
    mesh.f.swap(mesh.f_new);
    mesh.g.swap(mesh.g_new);
    std::fill(mesh.f_new.begin(), mesh.f_new.end(), 0.0);
    std::fill(mesh.g_new.begin(), mesh.g_new.end(), 0.0);
}

// ============================================================================
// EXPORTAÇÃO DIRETA DE IMAGENS PNG (usando stb_image_write)
// ============================================================================
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static inline uint8_t clamp8(double v) {
    if (v < 0.0) return 0;
    if (v > 255.0) return 255;
    return static_cast<uint8_t>(v + 0.5);
}

// Jet-like colormap simples: v in [0,1]
void jet_colormap(double v, uint8_t &r, uint8_t &g, uint8_t &b) {
    if (!std::isfinite(v)) v = 0.0;
    if (v < 0.0) v = 0.0;
    if (v > 1.0) v = 1.0;
    double rr = std::min(std::max(1.5 - std::fabs(4.0*v - 3.0), 0.0), 1.0);
    double gg = std::min(std::max(1.5 - std::fabs(4.0*v - 2.0), 0.0), 1.0);
    double bb = std::min(std::max(1.5 - std::fabs(4.0*v - 1.0), 0.0), 1.0);
    r = clamp8(rr * 255.0);
    g = clamp8(gg * 255.0);
    b = clamp8(bb * 255.0);
}

void write_images(const LatticeMesh& mesh, int step) {
    // ----- imagem de concentração (grayscale) -----
    std::string fnameC = "sim_conc_" + std::to_string(step) + ".png";
    std::vector<uint8_t> imgC(NX * NY * 3);
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            double c = mesh.C[mesh.idx(x,y)];
            if (!std::isfinite(c)) c = 0.0;
            if (c < 0.0) c = 0.0;
            if (c > 1.0) c = 1.0;
            uint8_t v = clamp8(c * 255.0);
            size_t p = 3 * (y * NX + x);
            imgC[p+0] = v;
            imgC[p+1] = v;
            imgC[p+2] = v;
        }
    }
    stbi_write_png(fnameC.c_str(), NX, NY, 3, imgC.data(), NX * 3);

    // ----- imagem de velocidade (colormap) -----
    double max_v = 0.0;
    for(int y=0;y<NY;++y){
        for(int x=0;x<NX;++x){
            int id = mesh.idx(x,y);
            if (mesh.mask[id] == 1) continue;
            double vx = mesh.ux[id];
            double vy = mesh.uy[id];
            double mag = std::sqrt(vx*vx + vy*vy);
            if (mag > max_v) max_v = mag;
        }
    }
    if (max_v <= 0.0) max_v = 1.0;

    std::string fnameV = "sim_vel_" + std::to_string(step) + ".png";
    std::vector<uint8_t> imgV(NX * NY * 3);

    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int id = mesh.idx(x,y);
            uint8_t r,g,b;
            if (mesh.mask[id] == 1) {
                r=g=b=0;
            } else {
                double vx = mesh.ux[id];
                double vy = mesh.uy[id];
                double mag = std::sqrt(vx*vx + vy*vy);
                double vn = mag / max_v; // 0..1
                jet_colormap(vn, r, g, b);
            }
            size_t p = 3*(y*NX + x);
            imgV[p+0] = r;
            imgV[p+1] = g;
            imgV[p+2] = b;
        }
    }
    stbi_write_png(fnameV.c_str(), NX, NY, 3, imgV.data(), NX * 3);

    std::cout << "Salvos: " << fnameC << " , " << fnameV << std::endl;
}

// ============================================================================
// MÉTRICAS: Contagem simples de dedos
// ============================================================================
int countFingersColumn(const LatticeMesh &mesh, int x_mid, double threshold=0.5) {
    int count = 0;
    bool inFinger = false;
    for(int y = 0; y < NY; ++y) {
        double C = mesh.C[mesh.idx(x_mid, y)];
        if (C > threshold) {
            if (!inFinger) {
                count++;
                inFinger = true;
            }
        } else {
            inFinger = false;
        }
    }
    return count;
}

int countConnectedComponents(const LatticeMesh &mesh, double threshold=0.5) {
    int nx = NX, ny = NY;
    std::vector<char> visited(nx*ny, 0);
    int components = 0;
    std::queue<std::pair<int,int>> q;
    for(int y=0;y<ny;++y){
        for(int x=0;x<nx;++x){
            int id = mesh.idx(x,y);
            if (visited[id]) continue;
            if (mesh.C[id] <= threshold || mesh.mask[id]==1) {
                visited[id] = 1;
                continue;
            }
            components++;
            q.push({x,y});
            visited[id] = 1;
            while(!q.empty()){
                auto p = q.front(); q.pop();
                int cx = p.first, cy = p.second;
                const int dx[4] = {1,-1,0,0};
                const int dy[4] = {0,0,1,-1};
                for(int d=0; d<4; ++d){
                    int nx2 = cx + dx[d];
                    int ny2 = cy + dy[d];
                    if (nx2<0||nx2>=nx||ny2<0||ny2>=ny) continue;
                    int id2 = mesh.idx(nx2, ny2);
                    if (visited[id2]) continue;
                    if (mesh.mask[id2]==1) { visited[id2]=1; continue; }
                    if (mesh.C[id2] > threshold) {
                        visited[id2]=1;
                        q.push({nx2,ny2});
                    } else {
                        visited[id2]=1;
                    }
                }
            }
        }
    }
    return components;
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    std::cout << "=== Simulação LBM: Instabilidade Saffman-Taylor (imagens PNG) ===" << std::endl;
    std::cout << "Dimensoes: " << NX << "x" << NY << std::endl;
    std::cout << "Viscosidade CO2: " << NU_CO2 << ", Razao M: " << VISC_RATIO << std::endl;

    LatticeMesh mesh;

    // 1. Configuração Inicial
    generate_porous_media(mesh);
    initialize(mesh);

    // 2. Loop Principal
    for(int step = 0; step <= MAX_ITER; ++step) {
        collide_and_stream(mesh);
        apply_boundary_conditions(mesh);
        update_buffers(mesh);

        // Atualiza campo C macroscópico a partir de g (e normaliza/clampa)
        double total_mass = 0.0;
        #pragma omp parallel for reduction(+:total_mass)
        for(int y=0;y<NY;++y){
            for(int x=0;x<NX;++x){
                size_t id = mesh.idx(x,y);
                if (mesh.mask[id]==1) {
                    mesh.C[id] = 0.0;
                    continue;
                }
                double Cmac = 0.0;
                for(int k=0;k<9;++k) {
                    Cmac += mesh.g[mesh.idx(x,y,k)];
                }
                if (!std::isfinite(Cmac)) Cmac = 0.0;
                if (Cmac < 0.0) Cmac = 0.0;
                if (Cmac > 1.0) Cmac = 1.0;
                mesh.C[id] = Cmac;
                total_mass += Cmac;
            }
        }

        if (step % 1000 == 0) {
            std::cout << "Step: " << step << " | Massa CO2 Total: " << total_mass << std::endl;
        }

        if (step % OUTPUT_FREQ == 0) {
            int x_mid = NX/2;
            int dedos_col = countFingersColumn(mesh, x_mid, 0.5);
            int comps = countConnectedComponents(mesh, 0.5);
            std::cout << "Step " << step << " | dedos(column) = " << dedos_col
                      << " | componentes = " << comps << std::endl;
            write_images(mesh, step);
        }
    }

    std::cout << "Simulação concluída." << std::endl;
    return 0;
}
