#include "VtkWriter.h"
#include <fstream>
#include <iomanip>


void VtkWriter::write_vtk_scalar(const std::string &filename, const LBMState &st, const std::vector<double> &field, const std::string &name) {
std::ofstream ofs(filename);
ofs<<"# vtk DataFile Version 3.0\n";
ofs<<name<<"\n";
ofs<<"ASCII\n";
ofs<<"DATASET STRUCTURED_POINTS\n";
ofs<<"DIMENSIONS "<<st.NX<<" "<<st.NY<<" 1\n";
ofs<<"ORIGIN 0 0 0\n";
ofs<<"SPACING 1 1 1\n";
ofs<<"POINT_DATA "<<st.NX*st.NY<<"\n";
ofs<<"SCALARS "<<name<<" double 1\n";
ofs<<"LOOKUP_TABLE default\n";
ofs<<std::setprecision(8);
for(size_t j=0;j<st.NY;++j) for(size_t i=0;i<st.NX;++i) {
ofs<<field[j*st.NX + i]<<"\n";
}
ofs.close();
}


void VtkWriter::write_vtk_vector(const std::string &filename, const LBMState &st, const std::vector<double> &fx, const std::vector<double> &fy, const std::string &name) {
std::ofstream ofs(filename);
ofs<<"# vtk DataFile Version 3.0\n";
ofs<<name<<"\n";
ofs<<"ASCII\n";
ofs<<"DATASET STRUCTURED_POINTS\n";
ofs<<"DIMENSIONS "<<st.NX<<" "<<st.NY<<" 1\n";
ofs<<"ORIGIN 0 0 0\n";
ofs<<"SPACING 1 1 1\n";
ofs<<"POINT_DATA "<<st.NX*st.NY<<"\n";
ofs<<"VECTORS "<<name<<" double\n";
ofs<<std::setprecision(8);
for(size_t j=0;j<st.NY;++j) for(size_t i=0;i<st.NX;++i) {
size_t idx = j*st.NX + i;
ofs<<fx[idx]<<" "<<fy[idx]<<" 0\n";
}
ofs.close();
}