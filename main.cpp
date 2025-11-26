// main.cpp
#include <iostream>
#include "Config.hpp"
#include "LatticeMesh.hpp"
#include "Boundary.hpp"
#include "Simulation.hpp"
#include "Utils.hpp"

int main() {
    std::cout << "=== Simulação LBM Modularizada ===" << std::endl;
    std::cout << "Dimensoes: " << NX << "x" << NY << std::endl;
    
    // 1. Prepara ambiente
    create_output_directories(); // Cria pastas concentracao/ e velocidade/

    // 2. Inicializa Malha e Geometria
    LatticeMesh mesh;
    generate_porous_media(mesh);
    initialize(mesh);

    // 3. Loop Principal
    for(int step = 0; step <= MAX_ITER; ++step) {
        collide_and_stream(mesh);
        apply_boundary_conditions(mesh);
        update_buffers(mesh);
        
        double total_mass = 0.0;
        update_macroscopics_C(mesh, total_mass); // Calcula C e massa

        if (step % 1000 == 0) {
            std::cout << "Step: " << step << " | Massa CO2: " << total_mass << std::endl;
        }

        if (step % OUTPUT_FREQ == 0) {
            int dedos = countFingersColumn(mesh, NX/2);
            int comps = countConnectedComponents(mesh);
            std::cout << "Step " << step << " | Dedos: " << dedos << " | Comps: " << comps << std::endl;
            
            // Salva nas subpastas criadas
            write_images(mesh, step);
        }
    }

    std::cout << "Fim." << std::endl;
    return 0;
}