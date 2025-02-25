#include <cmath>
#include <fstream>
#include <iomanip>
#include <bitset>
#include "utils/Random.hpp"
#include "SmoothOwen.hpp"

uint64_t CountScramblingCount(unsigned int Depth, unsigned int D)
{
    const uint64_t paramsPerTrees = (1u << Depth) - 1;
    uint64_t totalParams = 1;
    for (unsigned int d = 0; d < D; d++) totalParams *= (1 << paramsPerTrees);

    return totalParams;
}

void SetOwenScrambling(uint64_t idx, unsigned int Depth, std::vector<Params>& params)
{
    const uint64_t paramsPerTrees = (1u << Depth) - 1;
    const uint64_t bidx = idx;
    for (unsigned int d = 0; d < params.size(); d++)
    {
        for (unsigned int i = 0; i < paramsPerTrees; i++) params[d][i] = -1e10;

        uint64_t mask = (1 << (params.size() + 1)) - 1;
        uint64_t didx = idx & mask;

        unsigned int i = 0;
        while (didx)
        {
            bool digit = didx & 1;

            if (digit) params[d][i] = 1e10;

            i++;
            didx = didx >> 1;
        }

        idx = idx >> paramsPerTrees;
    }
}


SmoothOwenScrambling::SmoothOwenScrambling(unsigned int D, unsigned char Depth, double alph) :
        alpha(alph), Depth(Depth), thetas(D, Params({(1u << Depth) - 1}))
{ 
    for (unsigned int i = 0; i < D; i++) thetas[i].zeros();
}

uint32_t SmoothOwenScrambling::NumParamsPerDims() const
{
    return thetas[0].size();
}

void SmoothOwenScrambling::Export(const std::string& filename) const
{
    std::ofstream out(filename.c_str());
    out << std::setprecision(20) << std::fixed;

    for (unsigned int i = 0; i < thetas.size(); i++)
    {
        for (unsigned int d = 0; d < thetas[i].size(); d++)
        {
            out << thetas[i][{d}] << " ";
        }
        out << '\n';
    }
}

void SmoothOwenScrambling::ExportGrad(const std::string& filename, const BinaryArray& pts, unsigned int d)
{
    std::ofstream out(filename.c_str());
    out << std::noboolalpha << std::fixed;

    out << std::setprecision(20) << std::fixed;


#ifdef WITH_OMP
    #pragma omp parallel for
#endif
    for (unsigned int i = 0; i < pts.shape[0]; i++)
    {
        unsigned int select = 0;
        
        for (unsigned int depth = 0; depth < pts.shape[2]; depth++)
        {
            auto flip = DFF(pts[{i, d, depth}], thetas[d][select]);

            gradientsValues[d][select][i] = flip.second / (1 << (depth + 1));
            select             = 2 * select + pts[{i, d, depth}] + 1;
        }
    }

    for (unsigned int p = 0; p < thetas[d].size(); p++)
    {
        for (unsigned int pi = 0; pi < gradientsValues[d][p].size(); pi++)
        {
            out << gradientsValues[d][p][pi] << " ";
        }
        out << '\n';
    }
}

void SmoothOwenScrambling::Randomize(unsigned long long seed, unsigned long long channel)
{
    pcg32 pcg; pcg.seed(seed, channel);

    for (unsigned int d = 0; d < thetas.size(); d++)
    {
        for (unsigned int t = 0; t < thetas[d].size(); t++)
        {
            thetas[d][t] = static_cast<double>((pcg.nextUInt() & 1));
        }
    }
}

std::pair<double, double> SmoothOwenScrambling::DFF(bool bit, double param)
{
    const double  tanh = std::tanh(alpha * (param - 0.5));
    const double dtanh = alpha * (1 - tanh * tanh);

    param = 0.5 * (tanh + 1);
    return std::make_pair(
        (1. - bit) * param + bit * (1. - param), 
        0.5 * (1.0 - 2.0 * bit) * dtanh
    );
}
    
void SmoothOwenScrambling::forward(const BinaryArray& pts, PointArray& out)
{
    for (unsigned int d = 0; d < pts.shape[1]; d++)
    {

#ifdef WITH_OMP
        #pragma omp parallel for
#endif
        for (unsigned int i = 0; i < pts.shape[0]; i++)
        {
            unsigned int select = 0;
            out[{i, d}] = 0.;
            
            for (unsigned int depth = 0; depth < pts.shape[2]; depth++)
            {
                auto flip = DFF(pts[{i, d, depth}], thetas[d][select]);

                out[{i, d}]                  += flip.first  / (1 << (depth + 1));
                gradientsValues[d][select][i] = flip.second / (1 << (depth + 1));
                select             = 2 * select + pts[{i, d, depth}] + 1;
            }
        }
    }
}

void FillRandomLeftoverBits(PointArray& dest, unsigned int seed, unsigned int DepthMin, unsigned int DepthMax)
{
    const double min_ = std::pow(2.0, -(double)DepthMax);
    const double max_ = std::pow(2.0, -(double)DepthMin);

    const double diff = (max_ - min_);

    std::cout << "Filling random bit from: " << DepthMin << " to " << DepthMax << "(ie: " << min_ << "to " << max_ << ")" << std::endl;

    pcg32 rand(seed);
    for (unsigned int i = 0; i < dest.shape[0]; i++)
    {
        for (unsigned int j = 0; j < dest.shape[1]; j++)
        {
            dest[{i, j}] = diff * rand.nextDouble() + min_;
        }
    }
}

void ApplyLeftoverBits(PointArray& dest, const PointArray& bits)
{

#ifdef WITH_OMP
    #pragma omp parallel for
#endif
    for (unsigned int i = 0; i < dest.shape[0]; i++)
    {
        for (unsigned int j = 0; j < dest.shape[1]; j++)
        {
            dest[{i, j}] += bits[{i, j}];
        }
    }
}

    // Apply backpropagation given gradient of the loss to the points
void SmoothOwenScrambling::backward(const PointArray& gradients, double lr)
{   
    for (unsigned int d = 0; d < gradients.shape[1]; d++)
    {

#ifdef WITH_OMP
        #pragma omp parallel for
#endif
        for (unsigned int p = 0; p < thetas[d].size(); p++)
        {
            double gradientValue = 0.0;
            for (unsigned int pi = 0; pi < gradientsIndices[d][p].size(); pi++)
            {
                unsigned int pointIndex = gradientsIndices[d][p][pi];
                gradientValue += gradientsValues[d][p][pointIndex] * gradients[{pointIndex, d}];
            }
            thetas[d][p] -= lr * gradientValue;
        }
    }
}

void SmoothOwenScrambling::backwardStore(const PointArray& gradients, double lr, PointArray& grads)
{   
    for (unsigned int d = 0; d < gradients.shape[1]; d++)
    {
#ifdef WITH_OMP
        #pragma omp parallel for
#endif
        for (unsigned int p = 0; p < thetas[d].size(); p++)
        {
            grads[{d, p}] = 0.0;
            for (unsigned int pi = 0; pi < gradientsIndices[d][p].size(); pi++)
            {
                unsigned int pointIndex = gradientsIndices[d][p][pi];
                grads[{d, p}] += gradientsValues[d][p][pointIndex] * gradients[{pointIndex, d}];
            }
            thetas[d][p] -= lr * grads[{d, p}];
        }
    }
}

// Evaluate array
void SmoothOwenScrambling::evaluate(const BinaryArray& pts, PointArray& out, unsigned int override_depth)
{
    if (override_depth == 0)
        override_depth = pts.shape[2];
        
    const unsigned int depth_max = std::min(pts.shape[2], override_depth);

    for (unsigned int d = 0; d < pts.shape[1]; d++)
    {

#ifdef WITH_OMP
        #pragma omp parallel for
#endif
        for (unsigned int i = 0; i < pts.shape[0]; i++)
        {
            unsigned int select = 0;
            out[{i, d}] = 0.0;

            for (unsigned int depth = 0; depth < depth_max; depth++)
            {
                const bool   flipBit   = (thetas[d][select]  >= 0.5);
                const double fuzzyFlip = (flipBit - !flipBit) * 1e10;

                auto flip = DFF(pts[{i, d, depth}], fuzzyFlip);

                out[{i, d}] += flip.first / (1 << (depth + 1));
                select = 2 * select + pts[{i, d, depth}] + 1;
            }

            // Copy remaining bits 
            for (unsigned int depth = depth_max; depth < pts.shape[2]; depth++)
                out[{i, d}] += pts[{i, d, depth}] / (double)(1 << (depth + 1));
        }
    }
}

// Allocate gradients (and compute indices) for a special binary array
void SmoothOwenScrambling::AllocateGradients(const BinaryArray& pts)
{
    gradientsIndices.resize(pts.shape[1]);
    gradientsValues .resize(pts.shape[1]);

    for (unsigned int d = 0; d < pts.shape[1]; d++)
    {
        std::vector<std::vector<unsigned int>> dimIndices(thetas[d].size());
        std::vector<std::vector<double>>       dimValues (
            thetas[d].size(), std::vector<double>(pts.shape[0], 0.0)
        );

        for (unsigned int i = 0; i < pts.shape[0]; i++)
        {
            unsigned int select = 0;
            for (unsigned int depth = 0; depth < pts.shape[2]; depth++)
            {
                dimIndices[select].push_back(i);
                select = 2 * select + 1 + pts[{i, d, depth}];
            }
        }

        gradientsIndices[d] = std::move(dimIndices);
        gradientsValues [d] = std::move(dimValues);
    }
}
