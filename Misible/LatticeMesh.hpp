#ifndef LATTICEMESH_HPP
#define LATTICEMESH_HPP

#include <vector>
#include <cstddef>
#include "Config.hpp"

struct LatticeMesh {
    // Populações
    std::vector<double> f;
    std::vector<double> f_new;
    std::vector<double> g;
    std::vector<double> g_new;

    // Macroscópicos
    std::vector<double> rho;    
    std::vector<double> ux;     
    std::vector<double> uy;     
    std::vector<double> C;      
    std::vector<double> nu_loc; 

    // Geometria (0 = Fluido, 1 = Sólido)
    std::vector<int> mask;

    LatticeMesh(); // Construtor movido para o cpp ou inline (aqui farei inline pela simplicidade)

    inline size_t idx(int x, int y, int k = 0) const {
        return static_cast<size_t>(k) * (NX * NY) + static_cast<size_t>(y) * NX + static_cast<size_t>(x);
    }
};

// Implementação inline do construtor para facilitar
inline LatticeMesh::LatticeMesh() {
    size_t N = NX * NY;
    f.resize(N * 9, 0.0);
    f_new.resize(N * 9, 0.0);
    g.resize(N * 9, 0.0);
    g_new.resize(N * 9, 0.0);
    rho.resize(N, RHO0);
    ux.resize(N, 0.0);
    uy.resize(N, 0.0);
    C.resize(N, 0.0);
    nu_loc.resize(N, NU_CO2 * VISC_RATIO); 
    mask.resize(N, 0);
}

#endif