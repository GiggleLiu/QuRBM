#include <cmath>
#include <complex>

#include "cutils.h"

const double log2_=std::log(2.);

double lncoshd(double x){
    x=std::abs(x);
    if(x>12){
        return x-log2_;
    }
    else{
        return std::log(std::cosh(x));
    }
}

std::complex<double> lncoshc(std::complex<double> x){
    double xr=x.real();
    double xi=x.imag();
    std::complex<double> res=lncoshd(xr);
    res+=std::log(std::complex<double>(std::cos(xi),std::tanh(xr)*std::sin(xi)));
    return res;
}
