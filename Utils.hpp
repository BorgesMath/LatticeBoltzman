#ifndef UTILS_HPP
#define UTILS_HPP

#include "LatticeMesh.hpp"
#include <string>

// Cria os diretórios de saída
void create_output_directories();

// Salva imagens nas pastas respectivas
void write_images(const LatticeMesh& mesh, int step);

// Métricas
int countFingersColumn(const LatticeMesh &mesh, int x_mid, double threshold=0.5);
int countConnectedComponents(const LatticeMesh &mesh, double threshold=0.5);

#endif