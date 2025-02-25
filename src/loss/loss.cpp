#include "loss.hpp"

// Note : not the original version
//  - Uses a double* instead of std::vector<double> for density 
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-variable"
#include "semi_discrete_ot_2d/transport.h"
#pragma GCC diagnostic pop

#include "gbn/gbn.hpp"


#include <fstream>
#include <iomanip>

Density2D Density2D::LoadBMP(const char* filepath)
{
    std::ifstream file(filepath, std::ios::binary);
	std::vector<char> header(54);
	file.read(header.data(), 54); // Read BMP header

	int width, height;
	file.seekg(18, std::ios::beg);
	file.read(reinterpret_cast<char*>(&width), 4);
	file.read(reinterpret_cast<char*>(&height), 4); // Read dimensions

	std::vector<unsigned char> img((size_t)width * height * 3);
	file.seekg(54, std::ios::beg); // Go to the start of the pixel data
	file.read(reinterpret_cast<char*>(img.data()), img.size()); // Read pixels

    std::vector<double> density(width * height);
	double sum = 0;
	for (unsigned int i = 0; i < density.size(); i++) {
		density[i] = (img[0 + i * 3] + img[1 + i * 3] + img[1 + i * 3]) / 3;
		sum += density[i];
	}
	for (unsigned int i = 0; i < density.size(); i++) {
		density[i] *= width * width / sum;
	}

    std::cout << "Density loaded: " << width << " - " << filepath << std::endl;
    Density2D d;
    d.data = std::move(density);
    d.width  = width;
    d.height = height;

    return d;
}

// Global : 
//  Avoid reallocating point array
//  Creating a class...
//  But more importantly : memory leaks !
transport::OptimalTransport2D ot;
double W2Loss2D(const PointArray& array, PointArray& grad)
{
    using namespace transport;

    ot.V.vertices.resize(array.shape[0]);
    for (unsigned int i = 0; i < ot.V.vertices.size(); i++)
        ot.V.vertices[i] = Vector(array[{i, 0}], array[{i, 1}]);
    
    double squaredOTdist = ot.optimize(100); //100 Newton iterations max.
    
    double norm = 2. / (double) array.shape[0];
    for (unsigned int i = 0; i < ot.V.vertices.size(); i++)
    {
        const Vector centroid = ot.V.voronoi[i].centroid();
        const Vector gradV = norm * (ot.V.vertices[i] - centroid);
        grad[{i, 0}] = gradV.coords[0];
        grad[{i, 1}] = gradV.coords[1];
    }
    
    return squaredOTdist;
}
double  W2Loss1pt2D(const PointArray& array, PointArray& grad)
{
    double lossV = 0.0;
    for (unsigned int d = 0; d < array.shape[1]; d++)
    {
        lossV   += (array[{0, d}] - 0.5) * (array[{0, d}] - 0.5);
        grad[{0, d}] = 2 * (array[{0, d}] - 0.5);
    }
    return lossV;
}

// Global : 
//  Avoid reallocating point array
//  Creating a class...
//  But more importantly : memory leaks !
transport::OptimalTransport2D ot_density;
void SetNonUniformDensity2D(const Density2D& density)
{
    ot_density.setDensity(density.data, density.width);
}
double NonUniformW2Loss2D(const PointArray& array, PointArray& grad, const Density2D& density)
{
    using namespace transport;

    ot_density.V.vertices.resize(array.shape[0]);
    for (unsigned int i = 0; i < ot_density.V.vertices.size(); i++)
        ot_density.V.vertices[i] = Vector(array[{i, 0}], array[{i, 1}]);
    
    double squaredOTdist = ot_density.optimize(100); //100 Newton iterations max.
    
    double norm = 2. / (double) array.shape[0];
    for (unsigned int i = 0; i < ot_density.V.vertices.size(); i++)
    {
        Vector barycenter;
        ot_density.V.voronoi[i].weighted_area(const_cast<double*>(density.data.data()), density.width, barycenter);
        
        const Vector gradV = norm * (ot_density.V.vertices[i] - barycenter);
        grad[{i, 0}] = gradV.coords[0];
        grad[{i, 1}] = gradV.coords[1];
    }

    return squaredOTdist;
}

double GBN_Sigma = 1.0;
void SetGBNSigma(double sigma)
{
    GBN_Sigma = sigma;
}
double GBNLoss(const PointArray& array, PointArray& grad)
{
    double sigma = GBN_Sigma / std::pow(array.shape[0], 1.0 / (double)array.shape[1]);
    return gbn_toroidal_forward_backward(array, grad, sigma);
    return gbn_euclidean_forward_backward(array, grad, sigma);
}

#include "semi_discrete_ot_3d/geogram_wrapper.hpp"

Transport3D transport3D(std::string(DATA_PATH) + "/cube.geogram");
double W2Loss3D(const PointArray& array, PointArray& grad)
{
    double loss = transport3D.Compute(array.shape[0], array.data.data(), grad.data.data());

    const double norm = 2.0 / (double)array.shape[0];
#ifdef WITH_OMP
    #pragma omp parallel for
#endif
    for (unsigned int i = 0; i < grad.shape[0]; i++)
    {
        for (unsigned int j = 0; j < grad.shape[1]; j++)
        {
            grad[{i, j}] = norm * (array[{i, j}] - grad[{i, j}]);
        }
    }

    return loss;
}
