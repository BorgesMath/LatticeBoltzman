#ifndef LBM_CONSTANTS_H
#define LBM_CONSTANTS_H


#include <array>


constexpr int NX_DEF = 512; // default overwritten by params
constexpr int NY_DEF = 256;
constexpr double CS_SQ = 1.0/3.0;


constexpr std::array<int,9> CX = {0,1,0,-1,0,1,-1,-1,1};
constexpr std::array<int,9> CY = {0,0,1,0,-1,1,1,-1,-1};
constexpr std::array<double,9> W = {4.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0};
constexpr std::array<int,9> OPP = {0,3,4,1,2,7,8,5,6};


#endif // LBM_CONSTANTS_H