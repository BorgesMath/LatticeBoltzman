#ifndef VTKWRITER_H
#define VTKWRITER_H


#include <string>
#include "../include/LBMData.h"


class VtkWriter {
public:
static void write_vtk_scalar(const std::string &filename, const LBMState &st, const std::vector<double> &field, const std::string &name);
static void write_vtk_vector(const std::string &filename, const LBMState &st, const std::vector<double> &fx, const std::vector<double> &fy, const std::string &name);
};


#endif