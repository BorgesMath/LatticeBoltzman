#ifndef MAGNETIC_SOLVER_H
#define MAGNETIC_SOLVER_H


#include "../include/LBMData.h"


class MagneticSolver {
public:
double chi_max;
double H0_y;
double tolerance = 1e-6;
int max_iter = 5000;
MagneticSolver(double chi=2.0, double H0=0.01);
void update_permeability(LBMState &state);
void solve_potential(LBMState &state);
void compute_force(LBMState &state);
void apply_boundaries(LBMState &state);
};


#endif