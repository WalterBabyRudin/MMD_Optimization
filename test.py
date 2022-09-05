import numpy as np
import cvxpy as cp
from numpy.lib import utils
import torch
import scipy.spatial
import utils
import time


np.random.seed(1)

# Parameter Setting
d_hist = [50]#[10, 20, 50, 100, 200]
Maxiter = 50
rho = 1     # size of trust region
tol = 1e-6


for i in range(len(d_hist)):
    d = d_hist[i]
    N = 10*d

    x, y = utils.sample_generate(N,d)
    sigma = utils.kernelwidthPair(x,y)
    sigma = sigma/2

    sol_hist = np.load("sol_hist_d_50.npy")
    z = sol_hist[37,:]
    z_optimal = utils.MMD_grid_search(x, y, z, sigma, k=1)
    print(z_optimal)