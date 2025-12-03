#ifndef LBM_SOLVER_H
#define LBM_SOLVER_H


#include "../include/LBMData.h"
#include "MagneticSolver.h"


class LBMSolver {
public:
LBMParams params;
LBMState state;
LBMState next;
MagneticSolver mag;
LBMSolver(const LBMParams &p);
void initialize();
void step();
void run();
};


#endif