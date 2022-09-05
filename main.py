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

    sol_hist = np.zeros([Maxiter, d])


    start_time = time.time()
    # initialize variable selector
    z = np.ones([d,1])
    R = utils.MMD_stat_compute(x, y, z, sigma)
    k = d

    for iter in range(Maxiter):
        # Solve subproblem
        f_0, g_0, H_0 = utils.MMD_grad_compute(x, y, z, sigma)
        H_1 = utils.get_near_psd(-H_0)
        k = k-1
        z = utils.z_update(z,g_0,-H_1,rho,k=k)
        sol_hist[iter, :] = z
        # Compute Stat
        R_new = utils.MMD_stat_compute(x, y, z, sigma)
        Res = np.abs(R - R_new)
        print("Iter: ", iter, "MMD Stat: ", R_new, "Residual: ",Res)
        print(z.astype(int))
        R = R_new
        # Termination
        if (np.sum(z)<=3) or (R_new <= 1e-6):
            iter_final = iter
            break
    sol_hist = sol_hist[0:iter_final,:]
    np.save("sol_hist_d_"+str(d)+".npy", sol_hist)
        
    if (R_new <= 1e-6) and (iter_final>1):
        z = sol_hist[iter_final-1,:]
        z_optimal = utils.MMD_grid_search(x, y, z, sigma, k=1)
    else:
        z_optimal = utils.MMD_grid_search(x, y, z, sigma, k=1)
    # f_0, g_0, H_0 = utils.MMD_grad_compute(x, y, z, sigma)
    # H_1 = utils.get_near_psd(-H_0)
    # z = utils.z_update(z,g_0,-H_1,1.5,k=1)
    print(z_optimal)
    running_time = time.time() - start_time
    print("Running Time for d=",d," is ",running_time)
