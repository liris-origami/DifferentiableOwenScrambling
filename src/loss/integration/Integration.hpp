#pragma once

#include <utils/Random.hpp>

#include <utils/Array.hpp>
#include <fstream>
#include <vector>

template<typename Func>
class IntegrationTest
{
public:
    IntegrationTest(unsigned int Npts, unsigned int d, const char* file, unsigned long long seed = 0);
    IntegrationTest& NextBatch(unsigned int batchSize);

    void setSeed(uint64_t seed);

    double operator()(const PointArray& array, PointArray& grad) const;

    unsigned int IntegrandsCount() const { return integrandsCount; }

    const unsigned int Dim;
    const unsigned int NumParamsPerIntegrands;
private:
    mutable PointArray tmpGrad; // required by loss computation

    pcg32 random;

    unsigned int integrandsCount;
    unsigned int startBatch;
    unsigned int endBatch;

    std::vector<double> integrandsData;
};

template<typename Func>
IntegrationTest<Func>::IntegrationTest(
    unsigned int Npts, unsigned int d, 
    const char* file, unsigned long long seed) : 
        Dim(d), NumParamsPerIntegrands(Func::ComputeNumParams(d)), tmpGrad({Npts, d})
{ 
    random.seed(seed, 123456789);
    
    std::ifstream in(file);    
    if (in.is_open())
    {   
        while (in) 
        {
            integrandsData.resize(integrandsData.size() + NumParamsPerIntegrands + 1);

            in  >> integrandsData[integrandsData.size() - NumParamsPerIntegrands - 1];

            if (!in.good()) // Skip end of line in case ! (misread)
            {
                integrandsData.resize(integrandsData.size() - NumParamsPerIntegrands - 1);
                break;
            }
            Func::ReadData(in, Dim, &integrandsData[integrandsData.size() - NumParamsPerIntegrands]);   
        }
    }

    integrandsCount = integrandsData.size() / (NumParamsPerIntegrands + 1);
    std::cout << "Read: " << integrandsCount << " integrands" << std::endl;
    NextBatch(integrandsCount);
}

template<typename Func>
void IntegrationTest<Func>::setSeed(uint64_t seed)
{
    random.seed(seed, 123456789);
    NextBatch(integrandsCount);
}

template<typename Func>
IntegrationTest<Func>& IntegrationTest<Func>::NextBatch(unsigned int batchSize)
{
    if (integrandsCount <= 4 || batchSize >= integrandsCount - 4) // -4 : little margin 
                                          // So that the random works fine.
    {
        startBatch = 0; 
        endBatch = integrandsCount;
        return *this;
    }

    startBatch = random.nextUInt(integrandsCount - batchSize);
    endBatch   = startBatch + batchSize; 
    return *this;
}

template<typename Func>
double IntegrationTest<Func>::operator()(const PointArray& array, PointArray& grad) const
{
    const double invB  = 1.0 / (double)(endBatch - startBatch);
    const double invN  = 1.0 / (double)array.shape[0];
    const double inv2B = 2.0 * invB * invN;
    
    double loss = 0.;
    grad.zeros();

    for (unsigned int b = startBatch; b < endBatch; b++)
    {
        const double* dataStart = integrandsData.data() + b * (NumParamsPerIntegrands + 1);
        const double  gt = dataStart[0]; 

        double mcValue = 0.0;

#ifdef WITH_OMP
        #pragma omp parallel for reduction(+: mcValue)
#endif
        for (unsigned int i = 0; i < array.shape[0]; i++)
        {
            mcValue += Func::Evaluate(Dim, array.pointAt(i), dataStart + 1, tmpGrad.pointAt(i));
        }

        const double error = (mcValue * invN - gt);

#ifdef WITH_OMP
        #pragma omp parallel for
#endif
        for (unsigned int i = 0; i < array.shape[0]; i++)
        {
            for (unsigned int j = 0; j < Dim; j++)
            {
                grad[{i, j}] += inv2B * error * tmpGrad[{i, j}];
            }
        }

        loss += error * error;
    }

    return loss * invB;
}