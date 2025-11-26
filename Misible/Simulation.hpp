#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include "LatticeMesh.hpp"

void generate_porous_media(LatticeMesh& mesh);
void initialize(LatticeMesh& mesh);
void collide_and_stream(LatticeMesh& mesh);
void update_buffers(LatticeMesh& mesh);
void update_macroscopics_C(LatticeMesh& mesh, double& total_mass);

#endif