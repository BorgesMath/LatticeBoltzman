#include "Utils.hpp"
#include "Config.hpp"
#include <iostream>
#include <vector>
#include <queue>
#include <sys/stat.h> // Para mkdir no Linux
#include <sys/types.h>
#include <cmath>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void create_output_directories() {
    // Cria pastas modo Linux/Unix (0777 = permissão full)
    // Se estivesse no Windows usaria _mkdir
    mkdir("output", 0777);
    mkdir("output/concentracao", 0777);
    mkdir("output/velocidade", 0777);
    // std::cout << "Diretorios criados." << std::endl;
}

static inline uint8_t clamp8(double v) {
    if (v < 0.0) return 0; if (v > 255.0) return 255;
    return static_cast<uint8_t>(v + 0.5);
}

void jet_colormap(double v, uint8_t &r, uint8_t &g, uint8_t &b) {
    if (v < 0.0) v = 0.0; if (v > 1.0) v = 1.0;
    double rr = std::min(std::max(1.5 - std::fabs(4.0*v - 3.0), 0.0), 1.0);
    double gg = std::min(std::max(1.5 - std::fabs(4.0*v - 2.0), 0.0), 1.0);
    double bb = std::min(std::max(1.5 - std::fabs(4.0*v - 1.0), 0.0), 1.0);
    r = clamp8(rr * 255.0); g = clamp8(gg * 255.0); b = clamp8(bb * 255.0);
}

void write_images(const LatticeMesh& mesh, int step) {
    // 1. Concentração (Grayscale + Contour) -> output/concentracao/
    std::string fnameC = "output/concentracao/sim_conc_" + std::to_string(step) + ".png";
    std::vector<uint8_t> imgC(NX * NY * 3);
    
    // Parâmetro para linha de contorno
    const double CONTOUR_VAL = 0.5;

    #pragma omp parallel for
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            size_t id = mesh.idx(x,y);
            
            // Fundo Sólido = Preto
            if(mesh.mask[id] == 1) {
                size_t p = 3*(y*NX+x);
                imgC[p]=0; imgC[p+1]=0; imgC[p+2]=0;
                continue;
            }

            double c = mesh.C[id];
            
            // Normalização de segurança
            if (c < 0.0) c = 0.0; 
            if (c > 1.0) c = 1.0;

            uint8_t r, g, b;

            // --- Lógica de Cores ---
            if (SIM_MODE == MISCIBLE) {
                // Miscível: Escala de Cinza + Linha Vermelha
                uint8_t val = clamp8(c * 255.0);
                r = val; g = val; b = val;

                // Detecção de Borda Simples para o Contorno
                bool is_edge = false;
                // Checa direita
                if (x < NX-1) {
                    double c_right = mesh.C[mesh.idx(x+1, y)];
                    if ((c < CONTOUR_VAL && c_right >= CONTOUR_VAL) || 
                        (c >= CONTOUR_VAL && c_right < CONTOUR_VAL)) is_edge = true;
                }
                // Checa cima
                if (y < NY-1) {
                    double c_up = mesh.C[mesh.idx(x, y+1)];
                    if ((c < CONTOUR_VAL && c_up >= CONTOUR_VAL) || 
                        (c >= CONTOUR_VAL && c_up < CONTOUR_VAL)) is_edge = true;
                }

                if (is_edge) {
                    r = 255; g = 0; b = 0; // Contorno Vermelho
                }

            } else {
                // Imiscível: Azul (Fase 1) vs Vermelho (Fase 2)
                // C representa fração da Fase A (Vermelha)
                r = clamp8(c * 255.0);       // Quanto mais A, mais vermelho
                b = clamp8((1.0-c) * 255.0); // Quanto menos A, mais azul
                g = 0;
            }

            size_t p = 3 * (y * NX + x);
            imgC[p+0] = r; imgC[p+1] = g; imgC[p+2] = b;
        }
    }
    stbi_write_png(fnameC.c_str(), NX, NY, 3, imgC.data(), NX * 3);

    // 2. Velocidade (Jet) -> output/velocidade/
    double max_v = 0.0;
    // Achar maximo para normalizar
    for(int i=0; i<NX*NY; ++i) {
        if(mesh.mask[i]==0) {
            double mag = std::sqrt(mesh.ux[i]*mesh.ux[i] + mesh.uy[i]*mesh.uy[i]);
            if(mag > max_v) max_v = mag;
        }
    }
    if(max_v <= 1e-6) max_v = 1.0;

    std::string fnameV = "output/velocidade/sim_vel_" + std::to_string(step) + ".png";
    std::vector<uint8_t> imgV(NX * NY * 3);

    #pragma omp parallel for
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int id = mesh.idx(x,y);
            uint8_t r,g,b;
            if (mesh.mask[id] == 1) { 
                r=g=b=0; 
            } else {
                double mag = std::sqrt(mesh.ux[id]*mesh.ux[id] + mesh.uy[id]*mesh.uy[id]);
                jet_colormap(mag/max_v, r, g, b);
            }
            size_t p = 3*(y*NX + x);
            imgV[p+0] = r; imgV[p+1] = g; imgV[p+2] = b;
        }
    }
    stbi_write_png(fnameV.c_str(), NX, NY, 3, imgV.data(), NX * 3);
}

// --- MÉTRICAS ---

int countFingersColumn(const LatticeMesh &mesh, int x_mid, double threshold) {
    int count = 0;
    bool inFinger = false;
    for(int y = 0; y < NY; ++y) {
        // Se estivermos em modo imiscível, C já é a fração. 
        // Se miscível, C é a concentração. A lógica é a mesma.
        if (mesh.C[mesh.idx(x_mid, y)] > threshold) {
            if (!inFinger) { count++; inFinger = true; }
        } else { inFinger = false; }
    }
    return count;
}

int countConnectedComponents(const LatticeMesh &mesh, double threshold) {
    std::vector<char> visited(NX*NY, 0);
    int components = 0;
    std::queue<std::pair<int,int>> q;
    
    for(int y=0;y<NY;++y){
        for(int x=0;x<NX;++x){
            int id = mesh.idx(x,y);
            // Ignora se já visitado, se for fundo (C <= threshold) ou sólido
            if (visited[id] || mesh.C[id] <= threshold || mesh.mask[id]==1) { 
                visited[id]=1; 
                continue; 
            }
            
            // Nova componente encontrada
            components++;
            q.push({x,y}); 
            visited[id]=1;
            
            // BFS para preencher a componente
            while(!q.empty()){
                auto p = q.front(); q.pop();
                int cx = p.first, cy = p.second;
                const int dx[4]={1,-1,0,0}, dy[4]={0,0,1,-1};
                
                for(int d=0; d<4; ++d){
                    int nx2=cx+dx[d], ny2=cy+dy[d];
                    if(nx2>=0 && nx2<NX && ny2>=0 && ny2<NY){
                        int id2 = mesh.idx(nx2,ny2);
                        if(!visited[id2] && mesh.mask[id2]==0) {
                            visited[id2]=1;
                            if(mesh.C[id2] > threshold) {
                                q.push({nx2,ny2});
                            }
                        }
                    }
                }
            }
        }
    }
    return components;
}