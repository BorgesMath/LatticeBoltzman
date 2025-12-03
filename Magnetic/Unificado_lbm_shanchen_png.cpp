// lbm_shanchen_png.cpp
// D2Q9 Multicomponent Shan-Chen LBM with Darcy drag and magnetic forcing.
// Outputs PNG images (phi, phi with interface contour, speed) into ./output/
// Requires: stb_image_write.h in same folder.
// Compile: g++ -O3 -fopenmp -std=c++17 lbm_shanchen_png.cpp -o lbm_sim
// Run: ./lbm_sim [Nx] [Ny] [maxIter]

#include <bits/stdc++.h>
#include <omp.h>
#include <filesystem>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
using namespace std;
namespace fs = std::filesystem;

// --------------------------- Basic types ---------------------------
struct Vec2 { double x=0.0, y=0.0;
    Vec2(){} Vec2(double X,double Y):x(X),y(Y){}
    Vec2 operator+(const Vec2& b) const { return Vec2(x+b.x,y+b.y); }
    Vec2 operator-(const Vec2& b) const { return Vec2(x-b.x,y-b.y); }
    Vec2 operator*(double s) const { return Vec2(x*s,y*s); }
    Vec2 operator/(double s) const { return Vec2(x/s,y/s); }
    Vec2& operator+=(const Vec2& b){ x+=b.x; y+=b.y; return *this;}
    double dot(const Vec2& b) const { return x*b.x + y*b.y; }
    double norm2() const { return x*x + y*y; }
};
static inline Vec2 operator*(double s, const Vec2 &v){ return Vec2(v.x*s, v.y*s); }

// --------------------------- Lattice D2Q9 ---------------------------
const int Q = 9;
const int cx[Q] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
const int cy[Q] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
double w[Q] = {4.0/9.0, 1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0, 1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0};
const double cs2 = 1.0/3.0;
const double cs4 = cs2*cs2;

// --------------------------- Simulation parameters (default) ---------------------------
struct Params {
    int Nx = 400;
    int Ny = 200;
    int maxIter = 60000;
    int outputEvery = 1000;
    double dt = 1.0;
    double dx = 1.0;

    // Shan-Chen
    double G = 2.0;
    double tau1 = 0.7;
    double tau2 = 1.2;

    // porous medium
    double mu1 = 1.0;
    double mu2 = 1.0;
    double k1 = 1e4;
    double k2 = 1e4;

    // magnetic
    double mu0 = 1.0;
    double chi = 0.5;
    double H0x = 0.0, H0y = 0.02;
    double H_grad = 0.0;

    // density initialization
    double rho1_in = 0.02;
    double rho2_init = 1.0;
    double inlet_velocity = 0.01;
} P;

// --------------------------- Helpers ---------------------------
inline int idx(int x, int y, int Nx, int Ny){
    if(x<0) x += Nx; if(x>=Nx) x -= Nx;
    if(y<0) y += Ny; if(y>=Ny) y -= Ny;
    return x + y * Nx;
}

struct Field {
    int Nx, Ny, N;
    vector< vector<double> > f; // f[Q][N]
    Field(int Nx_, int Ny_):Nx(Nx_),Ny(Ny_),N(Nx_*Ny_), f(Q, vector<double>(N,0.0)){}
};

// --------------------------- Colormap (jet-like) ---------------------------
static inline uint8_t clamp255(double v){ if(v<0) v=0; if(v>1) v=1; return (uint8_t)round(v*255.0); }
void jet_colormap(double t, uint8_t &r, uint8_t &g, uint8_t &b){
    // t in [0,1]
    double tt = min(max(t,0.0),1.0);
    double rr = 1.5 - fabs(4.0*tt - 3.0);
    double gg = 1.5 - fabs(4.0*tt - 2.0);
    double bb = 1.5 - fabs(4.0*tt - 1.0);
    rr = min(max(rr,0.0),1.0);
    gg = min(max(gg,0.0),1.0);
    bb = min(max(bb,0.0),1.0);
    r = clamp255(rr); g = clamp255(gg); b = clamp255(bb);
}

// --------------------------- Pseudo-potential psi ---------------------------
inline double psi_of_rho(double rho){ return 1.0 - exp(-rho); }

// --------------------------- Magnetic field and grad H^2 ---------------------------
Vec2 magnetic_field_at(int x, int y, int Nx, int Ny, const Params &par){
    double gy = 0.0;
    if (par.H_grad != 0.0) gy = par.H_grad * ((double)y/(double)Ny - 0.5);
    return Vec2(par.H0x, par.H0y + gy);
}
Vec2 gradient_H2_at(int x, int y, int Nx, int Ny, const Params &par){
    auto Hxy = magnetic_field_at(x,y,Nx,Ny,par);
    auto Hxp = magnetic_field_at((x+1)%Nx,y,Nx,Ny,par);
    auto Hxm = magnetic_field_at((x-1+Nx)%Nx,y,Nx,Ny,par);
    auto Hyp = magnetic_field_at(x, min(y+1,Ny-1),Nx,Ny,par);
    auto Hym = magnetic_field_at(x, max(y-1,0),Nx,Ny,par);
    double H2_xp = Hxp.norm2(), H2_xm = Hxm.norm2(), H2_yp = Hyp.norm2(), H2_ym = Hym.norm2();
    double dHx = 0.5*(H2_xp - H2_xm);
    double dHy = 0.5*(H2_yp - H2_ym);
    return Vec2(dHx, dHy);
}

// --------------------------- PNG writer helpers ---------------------------
// create RGB image buffer from scalar field (ny x nx)
void write_png_field(const string &fname, const vector<double> &field, int nx, int ny, bool normalize=true){
    vector<unsigned char> img(nx*ny*3);
    double mn = 1e300, mx = -1e300;
    if (normalize) {
        for (int i=0;i<nx*ny;i++){ mn = min(mn, field[i]); mx = max(mx, field[i]); }
        if (mx <= mn){ mn = 0.0; mx = 1.0; }
    } else { mn = 0.0; mx = 1.0; }
    double invrange = 1.0 / (mx - mn);
    for (int j=0;j<ny;j++){
        for (int i=0;i<nx;i++){
            double v = field[i + j*nx];
            double t = (v - mn) * invrange;
            uint8_t r,g,b; jet_colormap(t,r,g,b);
            int k = 3*(i + j*nx);
            img[k+0] = r; img[k+1] = g; img[k+2] = b;
        }
    }
    stbi_write_png(fname.c_str(), nx, ny, 3, img.data(), nx*3);
}

// overlay interface contour at level (e.g., 0.5) on top of scalar RGB image, write PNG
void write_png_field_with_interface(const string &fname, const vector<double> &field, int nx, int ny, double level=0.5){
    // first create RGB from field as float colors
    vector<unsigned char> img(nx*ny*3);
    double mn = 1e300, mx = -1e300;
    for (int i=0;i<nx*ny;i++){ mn = min(mn, field[i]); mx = max(mx, field[i]); }
    if (mx <= mn){ mn = 0.0; mx = 1.0; }
    double invrange = 1.0/(mx-mn);
    for (int j=0;j<ny;j++){
        for (int i=0;i<nx;i++){
            double v = field[i + j*nx];
            double t = (v - mn) * invrange;
            uint8_t r,g,b; jet_colormap(t,r,g,b);
            int k = 3*(i + j*nx);
            img[k+0] = r; img[k+1] = g; img[k+2] = b;
        }
    }
    // detect contour by checking neighbor sign changes relative to level
    for (int j=0;j<ny;j++){
        for (int i=0;i<nx;i++){
            double v = field[i + j*nx];
            bool onContour = false;
            // 4-neighbors
            int inb[4][2] = {{i+1,j},{i-1,j},{i,j+1},{i,j-1}};
            for (int n=0;n<4;n++){
                int xi = inb[n][0], yj = inb[n][1];
                if (xi<0) xi += nx; if (xi>=nx) xi -= nx;
                if (yj<0) yj += ny; if (yj>=ny) yj -= ny;
                double vn = field[xi + yj*nx];
                if ((v - level) * (vn - level) < 0.0) { onContour = true; break; }
            }
            if (onContour){
                int k = 3*(i + j*nx);
                // paint red overlay (blend)
                img[k+0] = 255;
                img[k+1] = 40;
                img[k+2] = 40;
            }
        }
    }
    stbi_write_png(fname.c_str(), nx, ny, 3, img.data(), nx*3);
}

// --------------------------- Main LBM solver ---------------------------
int main(int argc, char** argv){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc > 1) P.Nx = atoi(argv[1]);
    if (argc > 2) P.Ny = atoi(argv[2]);
    if (argc > 3) P.maxIter = atoi(argv[3]);

    int Nx = P.Nx, Ny = P.Ny;
    int N = Nx*Ny;
    cout << "LBM PNG output: Nx="<<Nx<<" Ny="<<Ny<<" maxIter="<<P.maxIter<<"\n";

    Field f1(Nx,Ny), f2(Nx,Ny), f1_next(Nx,Ny), f2_next(Nx,Ny);
    vector<double> rho1(N,0.0), rho2(N,0.0);
    vector<Vec2> F1(N), F2(N), velocity(N,Vec2(0,0));

    // init
    for (int y=0;y<Ny;y++){
        for (int x=0;x<Nx;x++){
            int id = idx(x,y,Nx,Ny);
            rho2[id] = P.rho2_init;
            double seed = 0.0;
            if (y < Ny/10 && (x > Nx/4 && x < 3*Nx/4)) seed = P.rho1_in;
            rho1[id] = seed;
            for (int i=0;i<Q;i++){
                f1.f[i][id] = w[i] * rho1[id];
                f2.f[i][id] = w[i] * rho2[id];
            }
        }
    }

    double omega1 = 1.0 / P.tau1, omega2 = 1.0 / P.tau2;

    // make output dir
    fs::create_directories("output");

    for (int step=0; step<=P.maxIter; ++step){

        // macroscopic densities and momentum
        #pragma omp parallel for
        for (int y=0;y<Ny;y++){
            for (int x=0;x<Nx;x++){
                int id = idx(x,y,Nx,Ny);
                double r1=0.0, r2=0.0;
                Vec2 mom(0,0);
                for (int i=0;i<Q;i++){
                    r1 += f1.f[i][id];
                    r2 += f2.f[i][id];
                    mom.x += (f1.f[i][id] + f2.f[i][id]) * cx[i];
                    mom.y += (f1.f[i][id] + f2.f[i][id]) * cy[i];
                }
                rho1[id] = r1; rho2[id] = r2;
                double rho = r1 + r2;
                if (rho > 1e-14) velocity[id] = mom / rho;
                else velocity[id] = Vec2(0,0);
            }
        }

        // forces
        #pragma omp parallel for
        for (int y=0;y<Ny;y++){
            for (int x=0;x<Nx;x++){
                int id = idx(x,y,Nx,Ny);
                Vec2 Fsh1(0,0), Fsh2(0,0);
                double psi1 = psi_of_rho(rho1[id]);
                double psi2 = psi_of_rho(rho2[id]);
                for (int i=0;i<Q;i++){
                    int xn = (x + cx[i] + Nx) % Nx;
                    int yn = y + cy[i];
                    if (yn < 0) yn += Ny;
                    if (yn >= Ny) yn -= Ny;
                    int nid = idx(xn,yn,Nx,Ny);
                    double psi1_n = psi_of_rho(rho1[nid]);
                    double psi2_n = psi_of_rho(rho2[nid]);
                    Fsh1.x += w[i] * psi2_n * cx[i];
                    Fsh1.y += w[i] * psi2_n * cy[i];
                    Fsh2.x += w[i] * psi1_n * cx[i];
                    Fsh2.y += w[i] * psi1_n * cy[i];
                }
                Fsh1.x *= -P.G; Fsh1.y *= -P.G;
                Fsh2.x *= -P.G; Fsh2.y *= -P.G;

                Vec2 u = velocity[id];
                Vec2 Fd1 = Vec2( - (P.mu1 / P.k1) * u.x, - (P.mu1 / P.k1) * u.y );
                Vec2 Fd2 = Vec2( - (P.mu2 / P.k2) * u.x, - (P.mu2 / P.k2) * u.y );

                Vec2 gradH2 = gradient_H2_at(x,y,Nx,Ny,P);
                Vec2 Fmag = Vec2(0.5 * P.mu0 * P.chi * gradH2.x, 0.5 * P.mu0 * P.chi * gradH2.y);

                F1[id] = Fsh1 + Fd1 + Fmag;
                F2[id] = Fsh2 + Fd2;
            }
        }

        // collision + streaming
        #pragma omp parallel for
        for (int y=0;y<Ny;y++){
            for (int x=0;x<Nx;x++){
                int id = idx(x,y,Nx,Ny);
                double r1 = rho1[id], r2 = rho2[id];
                double rho = r1 + r2;
                Vec2 u = velocity[id];
                double u2 = u.norm2();
                for (int i=0;i<Q;i++){
                    double c_dot_u = cx[i]*u.x + cy[i]*u.y;
                    double feq1 = w[i] * r1 * (1.0 + 3.0*c_dot_u + 4.5*c_dot_u*c_dot_u - 1.5*u2);
                    double feq2 = w[i] * r2 * (1.0 + 3.0*c_dot_u + 4.5*c_dot_u*c_dot_u - 1.5*u2);

                    Vec2 Fi1 = F1[id];
                    double term1 = (cx[i]-u.x)*Fi1.x + (cy[i]-u.y)*Fi1.y;
                    double term2 = c_dot_u * (cx[i]*Fi1.x + cy[i]*Fi1.y);
                    double Si1 = w[i] * ( term1 / cs2 + term2 / cs4 );

                    Vec2 Fi2 = F2[id];
                    double term1b = (cx[i]-u.x)*Fi2.x + (cy[i]-u.y)*Fi2.y;
                    double term2b = c_dot_u * (cx[i]*Fi2.x + cy[i]*Fi2.y);
                    double Si2 = w[i] * ( term1b / cs2 + term2b / cs4 );

                    double f1_post = f1.f[i][id] - omega1 * ( f1.f[i][id] - feq1 ) + P.dt * Si1;
                    double f2_post = f2.f[i][id] - omega2 * ( f2.f[i][id] - feq2 ) + P.dt * Si2;

                    int xs = (x + cx[i] + Nx) % Nx;
                    int ys = y + cy[i];
                    if (ys < 0) ys += Ny;
                    if (ys >= Ny) ys -= Ny;
                    int idn = idx(xs, ys, Nx, Ny);
                    f1_next.f[i][idn] = f1_post;
                    f2_next.f[i][idn] = f2_post;
                }
            }
        }

        f1.f.swap(f1_next.f);
        f2.f.swap(f2_next.f);

        // inlet injection (bottom strip)
        int inletHeight = max(1, Ny/40);
        for (int y=0;y<inletHeight;y++){
            for (int x=Nx/4; x<3*Nx/4; ++x){
                int id = idx(x,y,Nx,Ny);
                double target_rho1 = max(rho1[id], 0.5 * P.rho2_init);
                double ux = 0.0, uy = P.inlet_velocity;
                Vec2 u_in(ux, uy);
                double u2 = u_in.norm2();
                for (int i=0;i<Q;i++){
                    double c_dot_u = cx[i]*u_in.x + cy[i]*u_in.y;
                    double feq1 = w[i] * target_rho1 * (1.0 + 3.0*c_dot_u + 4.5*c_dot_u*c_dot_u - 1.5*u2);
                    f1.f[i][id] = feq1;
                }
                rho1[id] = target_rho1;
            }
        }

        // recompute macroscopic quickly
        #pragma omp parallel for
        for (int y=0;y<Ny;y++){
            for (int x=0;x<Nx;x++){
                int id = idx(x,y,Nx,Ny);
                double r1=0.0, r2=0.0; Vec2 mom(0,0);
                for (int i=0;i<Q;i++){
                    r1 += f1.f[i][id];
                    r2 += f2.f[i][id];
                    mom.x += (f1.f[i][id] + f2.f[i][id]) * cx[i];
                    mom.y += (f1.f[i][id] + f2.f[i][id]) * cy[i];
                }
                rho1[id] = r1; rho2[id] = r2;
                double rho = r1 + r2;
                if (rho>1e-14) velocity[id] = mom / rho; else velocity[id] = Vec2(0,0);
            }
        }

        // output PNGs
        if (step % P.outputEvery == 0){
            cout << "Step " << step << " / " << P.maxIter << "\n";
            vector<double> phi(N), speed(N);
            double maxspeed = 1e-16;
            for (int i=0;i<N;i++){
                double sum = rho1[i] + rho2[i];
                phi[i] = (sum>0) ? rho1[i]/sum : 0.0;
                speed[i] = sqrt(velocity[i].norm2());
                if (speed[i] > maxspeed) maxspeed = speed[i];
            }
            // write phi (normalized 0..1)
            string fphi = "output/phi_" + to_string(step) + ".png";
            write_png_field(fphi, phi, Nx, Ny, true);
            // write phi with interface contour
            string fint = "output/phi_" + to_string(step) + "_interface.png";
            write_png_field_with_interface(fint, phi, Nx, Ny, 0.5);
            // write speed normalized by maxspeed
            vector<double> speed_norm(N);
            if (maxspeed < 1e-12) maxspeed = 1.0;
            for (int i=0;i<N;i++) speed_norm[i] = speed[i] / maxspeed;
            string fspd = "output/speed_" + to_string(step) + ".png";
            write_png_field(fspd, speed_norm, Nx, Ny, true);
        }
    } // end time loop

    cout << "Finished. PNGs in ./output/\n";
    return 0;
}
