#pragma once

#include <vector>
#include <cmath>

#include "utils/BinaryUtils.hpp"
#include "utils/Logger.hpp"

// Templated so the compiler can optimize 
// as most as possible (kernels, loop unrolling, ...)
template<unsigned int patchSize>
struct SSLoss
{
    SSLoss(double sigma) : 
        isigma2(-1. / (sigma * sigma)), energies(patchSize * patchSize)
    {

    }

    template<typename Func>
    double operator()(
        const Func& loss, 
        const std::vector<PointArray>& array, 
              std::vector<PointArray>& grad)
    {
        // Assumes all pointsets have the same shape
        const unsigned int N = array[0].shape[0];
        const unsigned int D = array[0].shape[1];

        // First compute errors. DO NOT Leave parallelization to energy computation (n is small and hence not well parallel !)

#ifdef WITH_OMP
        #pragma omp parallel for
#endif
        for (unsigned int i = 0; i < patchSize; i++)
        {
            for (unsigned int j = 0; j < patchSize; j++)
            {
                const unsigned int idx = j + i * patchSize;
                energies[idx] = loss(array[idx], grad[idx]);
            }
        }

        double energy = 0.;
#ifdef WITH_OMP
        #pragma omp parallel for reduction(+: energy)
#endif
        for (unsigned int i = 0; i < patchSize; i++)
        {
            for (unsigned int j = 0; j < patchSize; j++)
            {
                const unsigned int idx = j + i * patchSize;
                double weightFactor = 0.;

                for (unsigned int k = 0; k < patchSize; k++)
                {
                    for (unsigned int l = 0; l < patchSize; l++)
                    {
                        const unsigned int kdx = l + k * patchSize;

                        const double pixel_dist    = (i - k) * (i - k) + (j - l) * (j - l);
                        const double pixel_weight  = std::exp(pixel_dist * isigma2); 
                        const double energies_dist = (energies[idx] - energies[kdx]);

                        energy       +=  pixel_weight * energies_dist * energies_dist;
                        weightFactor +=  pixel_weight * energies_dist;
                    }
                }

                for (unsigned int n = 0; n < N; n++)
                {
                    for (unsigned int d = 0; d < D; d++)
                    {
                        // -1 : the loss should be maximized
                        grad[idx][{n, d}] *= -weightFactor;
                    }
                }
            }
        }

        return energy;
    }


    double isigma2;
    std::vector<double> energies;

};