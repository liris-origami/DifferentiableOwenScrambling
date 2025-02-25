#include "PCF.hpp"

#include <fstream>
#include <cmath>

PCF::PCF(
    unsigned int N, unsigned int D,
    double sigma, 
    double ra, double rb, unsigned int nbins,
    const std::string& targetFilename):
    N(N), D(D), tmpGrad({N, D}), sigma(sigma), ra(ra), rb(rb), nbins(nbins)
{
    std::ifstream file(targetFilename);
    if (file.is_open())
    {
        double tmp;
        while (file >> tmp) target.push_back(tmp);
    }

    if (target.size() != nbins)
    {
        std::cout << "[WARNING]: Number of bins different in target (target=" << target.size() << ", computed=" << nbins << ")" << std::endl;
    }
}

double PCF::operator()(const PointArray& pts, PointArray& grad, std::vector<double>& out) const
{
    static constexpr double invPI = 0.3183098861837907;
    static constexpr double    PI = 3.1415926535897932;
    static constexpr double  NORM = 0.3989422804014327; // / sqrt(2 * PI) round

    const double isigma2      = 1.0 / (sigma * sigma);
    const double gaussianNorm = NORM / sigma;

    out.resize(nbins);
    grad.zeros();

    double mse = 0.;
    for (unsigned int pcfid = 0; pcfid < nbins; pcfid++)
    {
        tmpGrad.zeros();

        double r = ra + pcfid * (rb - ra) / (double)nbins;
        double pcfValue = 0.0;

#ifdef WITH_OMP
        #pragma omp parallel for reduction(+: pcfValue)
#endif
        for (unsigned int i = 0; i < N; i++)
        {
            for (unsigned int j = i + 1; j < N; j++)
            {
                // Compute distance (TODO: outside this loop)
                double dist_squared = 0.0;
                for (unsigned int d = 0; d < D; d++)
                {
                    const double dx = (pts[{i, d}] - pts[{j, d}]);
                    dist_squared += dx * dx;
                }

                // Gaussian smooting
                const double dist = std::sqrt(dist_squared);
                const double dr   = (r - dist);
                double value = gaussianNorm * std::exp(-0.5 * dr * dr * isigma2); 
                
                // Append value
                pcfValue += value + value; // symetric
            
                // Grad of (r - ||x - y||)^2 = -1.0/(||x - y||) (x - y) * (r - ||x - y||)
                // Grad of value             = -0.5 * isigma2 * norm * value * (x - y) * (r - ||x - y||) / ||x - y|| 

                // No x0.5 in grad because it is summed twice. Signs cancels out
                for (unsigned int d = 0; d < D; d++)
                    tmpGrad[{i, d}] += gaussianNorm * isigma2 * value * (pts[{i, d}] - pts[{j, d}]) * dr / dist;
            }
        }

        // Normalization "constants"
        const double cov  = 1.0 - 2 * (2 * r * invPI) + r * r * invPI;
        const double norm = 1.0 / (2 * PI * r * cov) / (N * (N - 1));
        const double error = (norm * pcfValue - target[pcfid]);

        // Weighted MSE to emphasize on begining of the curve
        // const double weight = std::exp(-10.0 * r);
        const double weight = 1.0;
        mse += weight * error * error;
        
        // gradient of mse
#ifdef WITH_OMP
        #pragma omp parallel for
#endif
        for (unsigned int i = 0; i < N; i++)
        {
            for (unsigned int d = 0; d < D; d++)
            {
                grad[{i, d}] += 2 * weight * error * norm * tmpGrad[{i, d}];
            }
        }

        out[pcfid] = pcfValue * norm;
    }

    return mse / nbins;
}