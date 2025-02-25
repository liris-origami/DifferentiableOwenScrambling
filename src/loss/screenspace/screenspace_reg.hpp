#pragma once

#include <vector>
#include <cmath>

#include "utils/BinaryUtils.hpp"

inline double pixel_distance(unsigned int i, unsigned int j, unsigned int k, unsigned int l, unsigned int maxSize)
{
    // return (i - k) * (i - k) + (j - l) * (j - l);
    double diffX = std::abs((double)i - (double)k);
    double diffY = std::abs((double)j - (double)l);

    if (diffX > (maxSize / 2)) diffX = maxSize - diffX;
    if (diffY > (maxSize / 2)) diffY = maxSize - diffY;

    return diffX * diffX + diffY * diffY;
}

// Templated so the compiler can optimize 
// as most as possible (kernels, loop unrolling, ...)
template<unsigned int patchSize>
struct SSRegLoss
{
    // ass and aenergy should represent relative importance of ss loss and the energy
    SSRegLoss(double sigma, double ass = 1.0, double aenergy = 1.0) : 
         sigma (sigma), 
        isigma2(-1. / (sigma * sigma)), energies(patchSize * patchSize), 
        alpha_ss(ass), alpha_energy(aenergy)
    {
        energy_scaling = 0.0;
        ss_scaling = 1.0;
    }



    template<typename Func>
    void ComputeMaxEnergies(
        const Func& loss, 
        const std::vector<PointArray>& array, 
              std::vector<PointArray>& grad)
    {
        // Assumes all pointsets have the same shape
        // const unsigned int N = array[0].shape[0];
        // const unsigned int D = array[0].shape[1];


        // First compute errors. Leave parallelization to energy computation        
        double tot_energy = 0.0;

#ifdef WITH_OMP
        #pragma omp parallel for reduction(+: tot_energy)
#endif
        for (unsigned int i = 0; i < patchSize; i++)
        {
            for (unsigned int j = 0; j < patchSize; j++)
            {
                const unsigned int idx = j + i * patchSize;
                energies[idx] = loss(array[idx], grad[idx]);
                tot_energy   +=  energies[idx];
            }
        }

        // Aprox : sum_{i, j, k, l} exp(-||(i, j) - (k, l)||^2/s^2) = s * sqrt(2 * pi)
        // const double norm = sigma * std::sqrt(2 * 3.141592653589793238462);
        // const double max_ss_energy = norm * max_energy * max_energy;

        // First compute max of ss energy
        double tot_ss_energy = 0.0;

#ifdef WITH_OMP
        #pragma omp parallel for reduction(+: tot_ss_energy)
#endif 
        for (unsigned int i = 0; i < patchSize; i++)
        {
            for (unsigned int j = 0; j < patchSize; j++)
            {
                const unsigned int idx = j + i * patchSize;
                
                double local_ss = 0.;
                for (unsigned int k = 0; k < patchSize; k++)
                {
                    for (unsigned int l = 0; l < patchSize; l++)
                    {
                        const unsigned int kdx = l + k * patchSize;

                        const double pixel_dist    = pixel_distance(i, j, k, l, patchSize);
                        const double pixel_weight  = std::exp(pixel_dist * isigma2); 
                        const double energies_dist = (energies[idx] - energies[kdx]);
                        local_ss += pixel_weight * energies_dist * energies_dist;
                    }
                }
                tot_ss_energy += local_ss;
            }
        }

        std::cout << alpha_energy << ", " << tot_energy << std::endl;
        std::cout << alpha_ss << ", " << tot_ss_energy << std::endl;
        energy_scaling = alpha_energy / tot_energy;
            ss_scaling = alpha_ss     / tot_ss_energy;
   
        std::cout << "Computed: " << energy_scaling << ", " << ss_scaling << std::endl;
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

        // First compute errors. Leave parallelization to energy computation
        // Nevermind: do not, typically N << patchSize * patchSize
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

        double totalEnergy = 0.;
        double ssenergy    = 0.;
#ifdef WITH_OMP
        #pragma omp parallel for reduction(+: ssenergy,totalEnergy)
#endif
        for (unsigned int i = 0; i < patchSize; i++)
        {
            for (unsigned int j = 0; j < patchSize; j++)
            {
                const unsigned int idx = j + i * patchSize;
                double weightFactor = 0.;

                totalEnergy += energies[idx];
                for (unsigned int k = 0; k < patchSize; k++)
                {
                    for (unsigned int l = 0; l < patchSize; l++)
                    {
                        const unsigned int kdx = l + k * patchSize;

                        const double pixel_dist    = pixel_distance(i, j, k, l, patchSize);
                        const double pixel_weight  = std::exp(pixel_dist * isigma2); 
                        const double energies_dist = (energies[idx] - energies[kdx]);

                        ssenergy     +=  pixel_weight * energies_dist * energies_dist;
                        weightFactor +=  pixel_weight * energies_dist;
                    }
                }
                weightFactor *= 2.0;

                for (unsigned int n = 0; n < N; n++)
                {
                    for (unsigned int d = 0; d < D; d++)
                    {
                        // -1 : the loss should be maximized
                        grad[idx][{n, d}] = (energy_scaling - ss_scaling * weightFactor) * grad[idx][{n, d}];
                    }
                }
            }
        }

        Logger::Global().PushValue("energy"       , energy_scaling * totalEnergy);
        Logger::Global().PushValue("energy_base"  ,                  totalEnergy);
        Logger::Global().PushValue("ssenergy"     ,     ss_scaling *    ssenergy);
        return energy_scaling * totalEnergy - ss_scaling * ssenergy;
    }

    const double sigma;
    const double isigma2;
    std::vector<double> energies;

    const double alpha_ss;
    const double alpha_energy;

    double ss_scaling;
    double energy_scaling;

};
