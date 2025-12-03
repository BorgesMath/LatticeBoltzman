cat > src/main.cpp <<'EOF'
#include "../include/LBMData.h"
#include "LBMSolver.h"
#include <iostream>
#include <fstream>
#include <sstream>

static LBMParams read_params(const std::string &fname){
    LBMParams p;
    std::ifstream ifs(fname);
    if(!ifs) return p;
    std::string line;
    while(std::getline(ifs,line)){
        if(line.size()==0) continue; if(line[0]=='#') continue;
        std::istringstream ss(line);
        std::string key; if(!(ss>>key)) continue;
        if(key=="NX") ss>>p.NX;
        else if(key=="NY") ss>>p.NY;
        else if(key=="NSTEPS") ss>>p.NSTEPS;
        else if(key=="OUTPUT_EVERY") ss>>p.OUTPUT_EVERY;
        else if(key=="tau_red") ss>>p.tau_red;
        else if(key=="tau_blue") ss>>p.tau_blue;
        else if(key=="surface_tension_A") ss>>p.surface_tension_A;
        else if(key=="beta_recolor") ss>>p.beta_recolor;
        else if(key=="chi_max") ss>>p.chi_max;
        else if(key=="H0_y") ss>>p.H0_y;
        else if(key=="body_force") ss>>p.body_force;
        else if(key=="output_dir") ss>>p.output_dir;
    }
    return p;
}

int main(int argc, char**argv){
    std::string param_file = "params.ini";
    if(argc>1) param_file = argv[1];
    LBMParams params = read_params(param_file);
    LBMSolver solver(params);
    solver.run();
    return 0;
}
EOF
