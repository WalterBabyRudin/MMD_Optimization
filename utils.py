import numpy as np
import cvxpy as cp
from numpy.lib import utils
import torch
import scipy.spatial
import mosek


def get_near_psd(A):
    C = (A + A.T)/2
    eigval, eigvec = np.linalg.eig(C)
    eigval[eigval < 0] = 0

    return eigvec.dot(np.diag(eigval)).dot(eigvec.T)

def Pdist2(x, y):
    """compute the paired distance between x and y."""
    Pdist = scipy.spatial.distance.cdist(x,y,'sqeuclidean')

    # # x_norm = (x ** 2).sum(1).view(-1, 1)
    # # if y is not None:
    # #     y_norm = (y ** 2).sum(1).view(1, -1)
    # # else:
    # #     y = x
    # #     y_norm = x_norm.view(1, -1)
    # # Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    # Pdist[Pdist<0]=0
    return Pdist

def kernelwidthPair(x1, x2):
    '''Implementation of the median heuristic. See Gretton 2012
       Pick sigma such that the exponent of exp(- ||x-y|| / (2*sigma2)),
       in other words ||x-y|| / (2*sigma2),  equals 1 for the median distance x
       and y of all distances between points from both data sets X and Y.
    '''
    n, nfeatures = x1.shape
    m, mfeatures = x2.shape
    
    k1 = np.sum((x1*x1), 1)
    q = np.tile(k1, (m, 1)).transpose()
    del k1
    
    k2 = np.sum((x2*x2), 1)
    r = np.tile(k2, (n, 1))
    del k2
    
    h= q + r
    del q,r
    
    # The norm
    h = h - 2*np.dot(x1,x2.transpose())
    h = np.array(h, dtype=float)
    
    mdist = np.median([i for i in h.flat if i])

    return mdist

def sample_generate(N, d, std=0.8):
    # Generate N samples from mu and nu, respectively
    # mu: Gaussian distribution in R^d
    # nu: First entry is Laplace distribution with zero mean and standard deviation 0.8. Remaining is Gaussian
    # Input:
    #     N: sample size
    #     d: data dimension

    x = np.random.randn(N,d)
    
    y = np.random.randn(N,d)
    
    scale = np.sqrt(std**2/2)
    y1 = np.random.laplace(0, scale, N)
    y[:,0] = y1.reshape([-1,])

    return x, y

def sample_generate_Gaussian(N, d, std=0.4):
    # Generate N samples from mu and nu, respectively
    # mu: Gaussian distribution in R^d
    # nu: First entry is Laplace distribution with zero mean and standard deviation 0.8. Remaining is Gaussian
    # Input:
    #     N: sample size
    #     d: data dimension

    x = np.random.randn(N,d)
    
    y = np.random.randn(N,d)
    
    #scale = std
    y1 = np.random.randn(N,1) * std
    y[:,0] = y1.reshape([-1,])

    return x, y


def MMD_grad_compute(x, y, z, sigma):
    # Computer objective function, gradient, Hessian based on iteration point z
    # Input:
    #     x: N*d matrix
    #     y: N*d matrix
    #     z: d*1 variable selector, boolean-valued
    N,d = np.shape(x)
    indicator = z>0
    hatx = x[:, indicator.reshape([-1,])]
    haty = y[:, indicator.reshape([-1,])]

    Dxx = Pdist2(hatx, haty)
    Dyy = Pdist2(haty, haty)
    Dxy = Pdist2(hatx, haty)

    Kx = np.exp(-Dxx / (2*sigma))
    Ky = np.exp(-Dyy / (2*sigma))
    Kxy = np.exp(-Dxy / (2*sigma))

    f_0 = np.mean(Kx) + np.mean(Ky) - 2 * np.mean(Kxy)

    g_0 = np.zeros([d,1])
    H_0 = np.zeros([d,d])
    for i in range(N):
        for j in range(N):
            qxx = ((x[i,:] - x[j,:])**2).reshape([-1,1])
            qyy = ((y[i,:] - y[j,:])**2).reshape([-1,1])
            qxy = ((x[i,:] - y[j,:])**2).reshape([-1,1])

            K_qxx = Kx[i,j]*qxx
            K_qyy = Ky[i,j]*qyy
            K_qxy = Kxy[i,j]*qxy
            g_0 = g_0 - 0.5 * K_qxx - 0.5 * K_qyy + 2 * 0.5 * K_qxy

            K_qqxx = K_qxx * qxx.T
            K_qqyy = K_qyy * qyy.T
            K_qqxy = K_qxy * qxy.T

            H_0 = H_0 + 0.25 * K_qqxx + 0.25 * K_qqyy - 2 * 0.25 * K_qqxy


    g_0 = g_0/(N**2)
    H_0 = H_0/(N**2)


    return f_0, g_0, H_0

def MMD_stat_compute(x, y, z, sigma):
    # Computer MMD Stat
    # Input:
    #     x: N*d matrix
    #     y: N*d matrix
    #     z: d*1 variable selector, boolean-valued
    N,d = np.shape(x)

    z = z.reshape([-1,1])

    #print(int(z))
    #print(z == np.ones([d,1]))
    indicator = (z == np.ones([d,1]))
    #print(np.shape(indicator))
    hatx = x[:, indicator.reshape([-1,])]
    haty = y[:, indicator.reshape([-1,])]

    Dxx = Pdist2(hatx, haty)
    Dyy = Pdist2(haty, haty)
    Dxy = Pdist2(hatx, haty)

    Kx = np.exp(-Dxx / (2*sigma))
    Ky = np.exp(-Dyy / (2*sigma))
    Kxy = np.exp(-Dxy / (2*sigma))

    f_0 = np.mean(Kx) + np.mean(Ky) - 2 * np.mean(Kxy)

    return f_0


def z_update(z0,g,H,rho,k=3):
#def z_update(z0,g,H,k=3):
    # update variable selector by mixed integer QP
    z0 = z0.reshape([-1,])
    g = g.reshape([-1,])
    d = len(z0)
    z = cp.Variable(d, boolean=True)
    Obj = 1/2*cp.quad_form(z-z0, H) + cp.sum(cp.multiply(g, z))
    Constraint = [cp.norm(z-z0)<=rho, cp.sum(z)<=k, cp.sum(z)>=1]
    #Constraint = [cp.sum(z)<=k, cp.sum(z)>=1, cp.abs(z-z0)<=rho]
    prob = cp.Problem(cp.Maximize(Obj), Constraint)
    prob.solve(solver=cp.MOSEK)

    return z.value


def MMD_grid_search(x, y, z, sigma, k):
    N,d = np.shape(x)
    #print(z[np.nonzero(z)])
    i = np.nonzero(z)
    _, K = np.shape(i)
    trial_idx_hist = np.zeros([100,k])
    risk_hist = np.zeros(100)
    for trial in range(100):
        ix = np.random.choice(K, k, replace=False)
        trial_idx_hist[trial, :] = i[0][ix]
        z_ix = np.zeros(d)
        z_ix[ix] = 1
        f_0 = MMD_stat_compute(x, y, z_ix, sigma)
        risk_hist[trial] = f_0
    
    risk_idx = np.argmax(risk_hist)

    optimal_index = trial_idx_hist[risk_idx,:].astype(int)
    z_optimal = np.zeros(d)
    z_optimal[optimal_index.reshape([-1,])] = 1

    return z_optimal

