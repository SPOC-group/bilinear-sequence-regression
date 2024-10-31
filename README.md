# bilinear-sequence-regression
Code for the paper "Bilinear Sequence Regression: Model for Learning from Long Sequences of High-dimensional Tokens"

Read the paper here: [link](https://arxiv.org/pdf/2410.18858)

## Dependencies for Python
We use Python 3.10 with the following libraries
- Torch 1.12.1
- Numpy 1.23.2
- Pandas 1.4.4
- CVXPY 1.3.0
- Matplotlib 3.5.3

## Dependencies for Julia
We use Julia 1.11.1 with the following libraries
- CSV v0.10.14
- DataFrames v1.7.0
- HCubature v1.7.0
- NLsolve v4.5.1
- Plots v1.40.8
- Polynomials v4.0.11
- QuadGK v2.11.1
- Roots v2.2.1
- Statistics v1.11.1
- StatsPlots v0.15.7
- LinearAlgebra v1.11.0
- Random v1.11.0

## Scripts description

### paper_notebook.ipynb
This Jupyter notebook produces the figures in the paper. All the data is included in the repository.

### main.jl
This Julia code contains all the functions needed to obtain the theory curves in our paper. 

### finalplots.jl
This script uses the functions in `main.jl` to generate the data for the plots in the paper, saving them in hte folder `plots`.

### standard_rect.py
This code provides a minimal example of how we run standard Gradient Descent on the bilinear sequence regression model initialised in the prior. The data would be saved in 'standard'. For convenience the parameters $D$, $\rho$, $\beta$ and $\alpha$ are read from command line. We recommend to run this code in a cluster.

### averaged_rect.py
This code provides a minimal example of how we run Averaged Gradient Descent on the bilinear sequence regression model initialised in the prior. The data would be saved in 'averaged'. For convenience the parameters $D$, $\rho$, $\beta$ and $\alpha$ are read from command line. We recommend to run this code in a cluster.

### standard_rect_zero.py
This code is the same as `standard_rect.py` but with the initialisation of the model with very small norm.

### averaged_rect_zero.py
This code is the same as `averaged_rect.py` but with the initialisation of the model with very small norm.

### min_norm.py
This code provides a minimal example of how we find the Minimal Nuclear Norm Estimator on the bilinear sequence regression model. It's recomended to not use sizes larger than $D=20$. For convenience the parameters $D$, $\rho$, $\beta$ and $\alpha$ are read from command line.