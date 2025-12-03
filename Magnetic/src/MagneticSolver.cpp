#include "MagneticSolver.h"
if(frac<0) frac=0; if(frac>1) frac=1;
state.mu_rel[i] = 1.0 + chi_max * frac;
}
}


void MagneticSolver::apply_boundaries(LBMState &state) {
int NX = state.NX, NY = state.NY;
// Dirichlet top/bottom to impose approximate vertical H0
for(int x=0;x<NX;++x){
state.psi[state.idx(x,0)] = 0.0;
state.psi[state.idx(x,NY-1)] = -H0_y * (NY-1);
}
// Neumann left/right: copy neighbor
for(int y=0;y<NY;++y){
state.psi[state.idx(0,y)] = state.psi[state.idx(1,y)];
state.psi[state.idx(NX-1,y)] = state.psi[state.idx(NX-2,y)];
}
}


void MagneticSolver::solve_potential(LBMState &state) {
int NX = state.NX, NY = state.NY;
double omega = 1.7; // relaxation
for(int iter=0; iter<max_iter; ++iter){
double max_diff = 0.0;
// red-black SOR
for(int color=0;color<2;++color){
#pragma omp parallel for reduction(max:max_diff)
for(int y=1;y<NY-1;++y){
for(int x=1;x<NX-1;++x){
if(((x+y)&1) != color) continue;
size_t c = state.idx(x,y);
size_t e = state.idx(x+1,y);
size_t w = state.idx(x-1,y);
size_t n = state.idx(x,y+1);
size_t s = state.idx(x,y-1);
double mu_e = 2.0/(1.0/state.mu_rel[c] + 1.0/state.mu_rel[e]);
double mu_w = 2.0/(1.0/state.mu_rel[c] + 1.0/state.mu_rel[w]);
double mu_n = 2.0/(1.0/state.mu_rel[c] + 1.0/state.mu_rel[n]);
double mu_s = 2.0/(1.0/state.mu_rel[c] + 1.0/state.mu_rel[s]);
double numer = mu_e*state.psi[e] + mu_w*state.psi[w] + mu_n*state.psi[n] + mu_s*state.psi[s];
double denom = (mu_e + mu_w + mu_n + mu_s);
double psi_star = numer / denom;
double diff = fabs(psi_star - state.psi[c]);
state.psi[c] = (1.0 - omega) * state.psi[c] + omega * psi_star;
if(diff > max_diff) max_diff = diff;
}
}
}
apply_boundaries(state);
if(max_diff < tolerance) break;
}
}


void MagneticSolver::compute_force(LBMState &state) {
int NX = state.NX, NY = state.NY;
double mu0 = 1.0;
#pragma omp parallel for
for(int y=1;y<NY-1;++y){
for(int x=1;x<NX-1;++x){
size_t c = state.idx(x,y);
double Hx = -(state.psi[state.idx(x+1,y)] - state.psi[state.idx(x-1,y)]) * 0.5;
double Hy = -(state.psi[state.idx(x,y+1)] - state.psi[state.idx(x,y-1)]) * 0.5;
double Hsq = Hx*Hx + Hy*Hy;
double dphase_dx = (state.phase[state.idx(x+1,y)] - state.phase[state.idx(x-1,y)]) * 0.5;
double dphase_dy = (state.phase[state.idx(x,y+1)] - state.phase[state.idx(x,y-1)]) * 0.5;
double coeff = -0.5 * mu0 * Hsq * (chi_max / 2.0);
state.fx_mag[c] = coeff * dphase_dx;
state.fy_mag[c] = coeff * dphase_dy;
}
}
}