#include <CLI/App.hpp>
#include <CLI/Formatter.hpp>
#include <CLI/Config.hpp>

#include <iostream>
#include <vector>
#include <random>

#include "SmoothOwen/SmoothOwen.hpp"
#include "utils/BinaryUtils.hpp"
#include "utils/Logger.hpp"

#include "utils/ProgressBar.hpp"
#include "loss/loss.hpp"
#include "loss/integration/Integrands.hpp"
#include "loss/integration/Integration.hpp"

#ifndef DATA_PATH   
    #define DATA_PATH "./"
#endif

int main(int argc, char** argv)
{
    std::string out_prefix = "tmp_01";
    std::string sobol_file = std::string(DATA_PATH) + "/sobol2D.dat";
    
    unsigned long long seed = std::random_device{}();
    
    std::vector<unsigned int> Ns;
    std::vector<double> Ws;

    unsigned int D = 2;
    unsigned int Depth = 8;
    unsigned int Nits = 16384;
   
    unsigned int fillDepth = 0;
    double lr = 1e5;
    double alpha = 5.0;
    double alphaSchedule = 1.;
    double lrSchedule = 1.;

    double gbnSigma = 1.;

    CLI::App app;
    app.add_option("-n", Ns, "Number of points")->capture_default_str();
    app.add_option("-w", Ws, "Weight for loss of each pts")->capture_default_str();
    app.add_option("-d", D, "Number of dimensions")->capture_default_str();
    app.add_option("--depth", Depth, "Depth of scrambling")->capture_default_str();
    app.add_option("--nits", Nits, "Number of gradient descent iterations")->capture_default_str();
    app.add_option("--lr", lr, "Learning rate")->capture_default_str();
    app.add_option("--lrSchedule", lr, "Learning rate schedule")->capture_default_str();
    app.add_option("--alpha", alpha, "Derivative strenth at tipping point")->capture_default_str();
    app.add_option("--alphaSchedule", alphaSchedule, "Alpha factor applied at each iterations")->capture_default_str();

    app.add_option("--prefix", out_prefix, "Output prefix")->capture_default_str();
    
    app.add_option("--sobol", sobol_file, "Path to sobol points")->capture_default_str();
    app.add_option("--fill_depth", fillDepth, "Total depth, greater values than --depth will be filled at random")->capture_default_str();

    app.add_option("--gbn_sigma", gbnSigma, "GBN Sigma")->capture_default_str();

    CLI11_PARSE(app, argc, argv);

    SmoothOwenScrambling scrambling(D, Depth, alpha);
    scrambling.Randomize(seed);

    const unsigned int N = *std::max_element(Ns.begin(), Ns.end());
    PointArray  init({N, D});
    PointArray  pts({N, D});
    PointArray  grad({N, D});
    PointArray  leftover({N, D});
    BinaryArray binpts({N, D, Depth});

    // Load pts
    LoadPts(sobol_file.c_str(), init);

    float_to_bin(init, Depth, binpts);
    scrambling.AllocateGradients(binpts);

    SetGBNSigma(gbnSigma);
    auto loss = [&](const PointArray& array, PointArray& grad){
	    return W2Loss2D(array, grad);
    };

    if (fillDepth > Depth)
    {
        FillRandomLeftoverBits(leftover, seed + 7891011, Depth + 1, fillDepth);
    }


    scrambling.Randomize(seed, 1234);
    scrambling.evaluate(binpts, pts);
    WritePts((out_prefix + "_init.dat").c_str(), pts);
    
    Logger::Global().SetExpectedSize(Nits);

    // for (unsigned int i = 0; i < Nits; i++)
    for (unsigned int k : ProgressBar(Nits))
    {
        scrambling.forward(binpts, pts);
        if (fillDepth > Depth) ApplyLeftoverBits(pts, leftover);
        
        double l = 0;
        for (unsigned int i = 0; i < Ns.size(); i++)
        {
             pts.shape[0] = Ns[i];
            grad.shape[0] = Ns[i];

            double nl = loss(pts, grad);
            // trick to scale gradients: scale learning rate instead (avoid O(N*D) loop)
            // trick to sum gradients : apply multiple time backward pass !
            Logger::Global().PushValue("loss_" + std::to_string(Ns[i]), nl);
            Logger::Global().PushValue("loss_scaled_" + std::to_string(Ns[i]), nl * Ws[i]);
            scrambling.backward(grad, lr * Ws[i]);

            l += nl * Ws[i];
        }
        
        Logger::Global().PushValue("loss", l);
        scrambling.alpha *= alphaSchedule;
        lr = lr * lrSchedule;
    
        pts.shape[0] = N;
        grad.shape[0] = N;
    }

    Logger::Global().ExportCSV(out_prefix + "_log.tmp");
    WritePts(out_prefix + "_soft.dat", pts);
    WritePts(out_prefix + "_" + std::to_string(Depth + 1) + ".dat", pts);     

    scrambling.Export(out_prefix + "_tree_soft.txt");   
}
