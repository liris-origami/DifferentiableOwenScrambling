# Differentiable Owen Scrambling 

Repository for the paper : [Differentiable Owen Scrambling](https://dl.acm.org/doi/10.1145/3687764), SIGGRAPH ASIA 2024 ([pdf](https://perso.liris.cnrs.fr/david.coeurjolly/publication/owen-diff-24/owen-diff-24.pdf)).

``` bibtex
@article{owenDiff24,
  title    = {Differentiable Owen Scrambling},
  author   = {Bastien Doignies and David Coeurjolly and Nicolas Bonneel and  Julie Digne and Jean-Claude Iehl and Victor Ostromoukhov},
  year     = {2024},
  month    = dec,
  journal  = { {ACM} Transactions on Graphics (Proceedings of SIGGRAPH Asia)},
  doi      = {10.1145/3687764},
  volume   = 43,
  number   = 6
}
```

# Dependancies

The code only requires a compiler supporting C++17 (C++11 possible by removing constexpr) and OpenMP installed. If OmpenMP cannot be found or if it fails to compile, you may pass the `-DWITH_OMP=OFF` option to the cmake call.

# Building

The project is setup with cmake. The commands to run are the following: 

``` bash
mkdir build
cd build
cmake .. # eventually cmake .. -DWITH_OMP=OFF
make -j 8 
```

# Running the code 

Executable have common parameters:

* N: number of samples (replaced by Ns for progressive optimization)
* D: dimenions
* sobol: file to read initial samples from. Does not need to be from the Sobol' sequence or a (0, m, 2)-net in txt file. Only N * D coordinates are read from this file, therefore, at least the dimension should match with the parameter provided. Files can be found in `data/` 
* lr: learning rate
* alpha: smoothing parameter
* lrSchedule: learning rate factor. $lr_{i} = (lr_{Schedule})^{i} * lr$
* alphaSchedule: alpha factor. $\alpha_{i} = (\alpha_{Schedule})^{i} * \alpha$
* prefix: output directory + prefix. See below for more explaination on outputs of the code
* fill_depth: simulates the bottom of the owen tree by adding small random noise to each points

Loss parameters:

* db: path to intrand database (required for integration error optimization), files are provided in `data/`
* batchSize: batch size for integration error optimization. 
* target: targetPCF
* sigmaGBN: sigma value for GBN. The `actual` gbn factor is $\sigma_{GBN} * N^{-2/D}$

The scripts outputs multiple file:

* {prefix}_k.dat: points after optimization after applying depth k.
* {prefix}_soft.dat: points after optimization where the tree is not binarized
* {prefix}_init.dat: initial points (after the starting random tree is applied, but before optimization)
* {prefix}_soft_tree.txt: floating point values stored in the tree
where {prefix} is the parameter passed to the script

# Plotting points 

A simple python script is provided in order to plot the points. It requires numpy and matplotlib. 

`python plot.py pts.dat`

# Example commands (Fig. 3)

To reproduce the examples from Fig. 3 in the paper, you may use the following commands inside the `build` directory: 

* "Ours/W2": `./DifferentiableOwenW2 -n 1024 -d 2 --nits 128 --lr 1e5 --depth 16 --fill_depth 32 --prefix outw2`
* "Ours/GBN": `./DifferentiableOwenGBN -n 1024 -d 2 --nits 256 --lr 1e5 --depth 16 --fill_depth 32 --gbn_sigma 0.5 --prefix outgbn`
* "Ours/Integration": `./DifferentiableOwenDiffInt -n 1024 -d 2 --nits 128 --lr 1e5 --depth 16 --fill_depth 32 --prefix outint`
* "Ours/l2 PCF" (LDBN target by default): `./DifferentiableOwenpcf -n 1024 -d 2 --nits 128 --lr 1e5 --depth 16 --fill_depth 32 --prefix outpcf`

The pointset can be visualized using the command:

`python ../plot.py {prefix}_17.dat`

Where prefix is one of the prefix option that are specified in the command (ie. one of: outw2, outgbn, outint, outpcf). 