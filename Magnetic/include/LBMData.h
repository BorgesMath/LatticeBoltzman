#ifndef LBM_DATA_H
#define LBM_DATA_H


#include <vector>
#include <string>
#include "LBMConstants.h"


struct LBMParams {
int NX = NX_DEF;
int NY = NY_DEF;
int NSTEPS = 10000;
int OUTPUT_EVERY = 500;
double tau_red = 0.6;
double tau_blue = 1.2;
double surface_tension_A = 0.002;
double beta_recolor = 0.7;
double chi_max = 2.0;
double H0_y = 0.01;
double body_force = 1e-6;
std::string output_dir = "output";
};


struct LBMState {
int NX, NY;
size_t N;
std::vector<double> f_red; // size 9*N
std::vector<double> f_blue;
std::vector<double> rho;
std::vector<double> ux;
std::vector<double> uy;
std::vector<double> phase; // -1..1
std::vector<double> psi;
std::vector<double> mu_rel;
std::vector<double> fx_mag;
std::vector<double> fy_mag;


LBMState(int nx=NX_DEF, int ny=NY_DEF): NX(nx), NY(ny) {
N = static_cast<size_t>(NX)*NY;
f_red.assign(9*N, 0.0);
f_blue.assign(9*N, 0.0);
rho.assign(N, 1.0);
ux.assign(N, 0.0);
uy.assign(N, 0.0);
phase.assign(N, -1.0);
psi.assign(N, 0.0);
mu_rel.assign(N, 1.0);
fx_mag.assign(N, 0.0);
fy_mag.assign(N, 0.0);
}


inline size_t idx(int x, int y) const { return static_cast<size_t>(y)*NX + x; }
};


#endif // LBM_DATA_H