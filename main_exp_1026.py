import numpy as np
import cvxpy as cp
from numpy.lib import utils
import torch
import scipy.spatial
import utils
import time


def exp_kernel_selector(a_xx, a_yy, a_xy, K=2):
    """
    Return optimal selector from Gaussian kernel based on optimization
         a_xx: N2*D data matrix
         a_yy: N2*D data matrix
         a_xy: N2*D data matrix
            K: number of features
       Output:
            z: optimal selector (binary variable)
    """
    M = 1

    N2, D = np.shape(a_xx)
    z = cp.Variable(D, boolean=True)
    e_xx = cp.Variable([N2, D+1])
    e_yy = cp.Variable([N2, D+1])
    e_xy = cp.Variable([N2, D+1])

    obj = cp.Minimize(-cp.sum(e_xx[:,D])-cp.sum(e_yy[:,D]) + 2*cp.sum(e_xy[:,D]))

    constraints = [cp.sum(z) <= K, e_xx[:,0] == 1, e_yy[:,0] == 1, e_xy[:,0] == 1]

    for i in range(N2):
        for j in range(D):
            constraints += [e_xx[:,j+1] - e_xx[:,j] <= M*z[j]]
            constraints += [e_xx[:,j+1] - e_xx[:,j] >= -M*z[j]]

            constraints += [e_yy[:,j+1] - e_yy[:,j] <= M*z[j]]
            constraints += [e_yy[:,j+1] - e_yy[:,j] >= -M*z[j]]

            constraints += [e_xy[:,j+1] - e_xy[:,j] <= M*z[j]]
            constraints += [e_xy[:,j+1] - e_xy[:,j] >= -M*z[j]]

            constraints += [e_xx[:,j+1] - e_xx[:,j]*a_xx[:,j] <= M*(1-z[j])]
            constraints += [e_xx[:,j+1] - e_xx[:,j]*a_xx[:,j] >= -M*(1-z[j])]

            constraints += [e_yy[:,j+1] - e_yy[:,j]*a_yy[:,j] <= M*(1-z[j])]
            constraints += [e_yy[:,j+1] - e_yy[:,j]*a_yy[:,j] >= -M*(1-z[j])]

            constraints += [e_xy[:,j+1] - e_xy[:,j]*a_xy[:,j] <= M*(1-z[j])]
            constraints += [e_xy[:,j+1] - e_xy[:,j]*a_xy[:,j] >= -M*(1-z[j])]



    prob = cp.Problem(obj, constraints)
    prob.solve(cp.GUROBI, verbose=True)








# Parameter Setting
d_hist = [10]#[10, 20, 50, 100, 200]


for i in range(len(d_hist)):
    d = d_hist[i]
    N = d

    x, y = utils.sample_generate(N,d)
    sigma = utils.kernelwidthPair(x,y)
    sigma = sigma/2

    a_xx,a_yy,a_xy = utils.MMD_exp_coeff_prepare(x, y, sigma)
    exp_kernel_selector(a_xx, a_yy, a_xy)
