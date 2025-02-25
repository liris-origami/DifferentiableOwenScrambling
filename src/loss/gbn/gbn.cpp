#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

#include "utils/Array.hpp"

template<typename FloatType> 
inline FloatType toroidal_warp_unsigned(FloatType a)
{
    const bool warp = (a > 0.5);
    return warp * (1. - a) + !warp * a;
}

template<typename FloatType> 
inline FloatType toroidal_warp_signed(FloatType a)
{
    const FloatType A = std::abs(a);

    const bool neg  = a < 0.0;
    const bool warp = A > 0.5;

    const bool outputSignBit = (neg * !warp) + (!neg * warp);
    const int  outputSign = 1 - 2 * !outputSignBit;

    return outputSign * (warp * (1 - A) + !warp * A);
}

double gbn_toroidal_forward_backward(const PointArray& array, PointArray& grad, double sigma)
{
    const double PI = 3.141592653589793238462;

    const unsigned int N  = array.shape[0];
    const unsigned int D  = array.shape[1];
    const double invSigma = - 1.0 / (2.0 * sigma * sigma); 
    const double norm     = PI * sigma * sigma / (2.0 * N);
    const double invN     = 1.0 / N;

    double rloss = 0.;

    grad.zeros();

#ifdef WITH_OMP
    #pragma omp parallel for reduction(+: rloss)
#endif
    for (unsigned int k = 0; k < N; k++)
    {
        for (unsigned int l = 0; l < N; l++)
        {
            double dist = 0.;
            for (unsigned int d = 0; d < D; d++)
            {
                double dx = toroidal_warp_unsigned<double>(std::abs(array[{k, d}] - array[{l, d}]));
                dist += dx * dx;
            }
            
            const double weight = std::exp(invSigma * dist) * invN;
            for (unsigned int d = 0; d < D; d++)
                grad[{k, d}] += weight * toroidal_warp_signed<double>(array[{k, d}] - array[{l, d}]);

            rloss += weight * norm;
        }   
    }

    return rloss;
}

double gbn_euclidean_forward_backward(const PointArray& array, PointArray& grad, double sigma)
{
    const double PI = 3.141592653589793238462;

    const unsigned int N  = array.shape[0];
    const unsigned int D  = array.shape[1];
    const double invSigma = - 1.0 / (2.0 * sigma * sigma); 
    const double norm     = PI * sigma * sigma / (2.0 * N);
    const double invN     = 1.0 / N;

    double rloss = 0.;

    grad.zeros();

    #pragma omp parallel for reduction(+: rloss)
    for (unsigned int k = 0; k < N; k++)
    {
        for (unsigned int l = 0; l < N; l++)
        {
            double dist = 0.;
            for (unsigned int d = 0; d < D; d++)
            {
                double dx = (array[{k, d}] - array[{l, d}]);
                dist += dx * dx;
            }
            
            const double weight = std::exp(invSigma * dist) * invN;
            for (unsigned int d = 0; d < D; d++)
                grad[{k, d}] += weight * (array[{l, d}] - array[{k, d}]);

            rloss += weight;
        }   
    }

    return rloss * norm;
}