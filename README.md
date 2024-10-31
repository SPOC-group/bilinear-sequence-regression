# bilinear-sequence-regression
Code for the paper "Bilinear Sequence Regression: Model for Learning from Long Sequences of High-dimensional Tokens"

Read the paper here: [link](https://arxiv.org/pdf/2410.18858)

## Dependencies for Python

- Torch 1.12.1
- Numpy 1.23.2
- Pandas 1.4.4
- CVXPY 1.3.0
- Matplotlib 3.5.3

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