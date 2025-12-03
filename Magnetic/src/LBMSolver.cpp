#include "LBMSolver.h"
#include "VtkWriter.h"
#include <cmath>
#include <filesystem>
#include <iostream>


LBMSolver::LBMSolver(const LBMParams &p): params(p), state(p.NX,p.NY), next(p.NX,p.NY), mag(p.chi_max,p.H0_y) {
}


void LBMSolver::initialize() {
int NX = state.NX, NY = state.NY;
// Inicializa densidades red/blue com interface vertical em NX/2 + pequena perturbação senoidal
for(int y=0;y<NY;++y){
for(int x=0;x<NX;++x){
size_t id = state.idx(x,y);
double xint = NX/2 + 3.0 * sin(2.0*M_PI*y/16.0); // pequena perturba
if(x < xint) { // esquerda: fluido paramagnético (red)
state.phase[id] = 1.0;
} else {
state.phase[id] = -1.0;
}