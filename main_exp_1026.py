# from msilib.schema import Binary
# from telnetlib import BINARY
from statistics import mode
import numpy as np
import cvxpy as cp
from numpy.lib import utils
import torch
import scipy.spatial
import utils
import time
from gurobipy import GRB

import gurobipy as gp
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
    #z = cp.Variable(D)
    z = cp.Variable(D, boolean=True)
    e_xx = cp.Variable([N2, D+1])
    e_yy = cp.Variable([N2, D+1])
    e_xy = cp.Variable([N2, D+1])

    obj = cp.Minimize(-cp.sum(e_xx[:,D])-cp.sum(e_yy[:,D]) + 2*cp.sum(e_xy[:,D]))

    constraints = [cp.sum(z) <= K, e_xx[:,0] == 1, e_yy[:,0] == 1, e_xy[:,0] == 1, z >= 0, z<= 1]

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

def GRB_exp_kernel_selector(a_xx, a_yy, a_xy, z0, K=5):
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


    # create gurobi model 
    model = gp.Model("Exponential Kernel Selection")

    z = model.addVars(D, vtype=GRB.BINARY)
    e_xx = model.addVars(N2, D+1)
    e_yy = model.addVars(N2, D+1)
    e_xy = model.addVars(N2, D+1)

    MMD_obj = -e_xx.sum() - e_yy.sum() + 2*e_xy.sum()

    model.addConstr(z.sum() <= K)
    #print(e_xx)
    for i in range(N2):
        model.addConstr(e_xx[i,0] == 1)
        model.addConstr(e_yy[i,0] == 1)
        model.addConstr(e_xy[i,0] == 1)

        for d in range(D):
            model.addConstr((z[d] == 0) >> (e_xx[i,d+1] == e_xx[i,d]))
            model.addConstr((z[d] == 0) >> (e_yy[i,d+1] == e_yy[i,d]))
            model.addConstr((z[d] == 0) >> (e_xy[i,d+1] == e_xy[i,d]))

            model.addConstr((z[d] == 1) >> (e_xx[i,d+1] == a_xx[i,d] * e_xx[i,d]))
            model.addConstr((z[d] == 1) >> (e_yy[i,d+1] == a_yy[i,d] * e_yy[i,d]))
            model.addConstr((z[d] == 1) >> (e_xy[i,d+1] == a_xy[i,d] * e_xy[i,d]))

    z.start = z0
    model.setObjective(MMD_obj, GRB.MINIMIZE)
    model.optimize()

def GRB_exp_kernel_selector_efficient(a_xx, a_yy, a_xy, z0, K=5):
    """
    Return optimal selector from Gaussian kernel based on optimization
         a_xx: N2*D data matrix
         a_yy: N2*D data matrix
         a_xy: N2*D data matrix
            K: number of features
       Output:
            z: optimal selector (binary variable)
    """

    N2, D = np.shape(a_xx)


    # create gurobi model 
    model = gp.Model("Exponential Kernel Selection")

    z = model.addVars(D, vtype=GRB.BINARY)
    e_xx = model.addVars(N2, D+1)
    e_yy = model.addVars(N2, D+1)
    e_xy = model.addVars(N2, D+1)

    MMD_obj = -e_xx.sum() - e_yy.sum() + 2*e_xy.sum()

    model.addConstr(z.sum() <= K)
    #print(e_xx)
    for i in range(N2):
        model.addConstr(e_xx[i,0] == 1)
        model.addConstr(e_yy[i,0] == 1)
        model.addConstr(e_xy[i,0] == 1)

        for d in range(D):

            if np.abs(a_xx[i,d] - 1) <= 0.1:
                model.addConstr(e_xx[i,d+1] == e_xx[i,d])
            else:
                model.addConstr((z[d] == 0) >> (e_xx[i,d+1] == e_xx[i,d]))
                model.addConstr((z[d] == 1) >> (e_xx[i,d+1] == a_xx[i,d] * e_xx[i,d]))
            
            if np.abs(a_yy[i,d] - 1) <= 0.1:
                model.addConstr(e_yy[i,d+1] == e_yy[i,d])
            else:
                model.addConstr((z[d] == 0) >> (e_yy[i,d+1] == e_yy[i,d]))
                model.addConstr((z[d] == 1) >> (e_yy[i,d+1] == a_yy[i,d] * e_yy[i,d]))
            
            if np.abs(a_xy[i,d] - 1) <= 0.1:
                model.addConstr(e_xy[i,d+1] == e_xy[i,d])
            else:
                model.addConstr((z[d] == 0) >> (e_xy[i,d+1] == e_xy[i,d]))
                model.addConstr((z[d] == 1) >> (e_xy[i,d+1] == a_xy[i,d] * e_xy[i,d]))
        
    z.start = z0
    model.setObjective(MMD_obj, GRB.MINIMIZE)
    model.optimize()






def inner_product_compute(X, Y):
    """
    Compute sum_{i,j}X_i[k]Y_j[k] for k = 1,...,D
        Input:
            X: n*D data matrix
            Y: m*D data matrix
       Output:
            z: D-dim array
    """
    n, D = np.shape(X)
    m, _ = np.shape(Y)
    z = np.zeros(D)

    for i in range(n):
        for j in range(m):
            z = z + X[i,:]*Y[j,:]

    # for k in range(D):
    #     for i in range(n):
    #         for j in range(m):
    #             z[k] = z[k] + X[i,k]*Y[j,k]
    return z

def linear_kernel_selector(X, Y, K=5):
    """
    Return optimal selector from linear kernel based on optimization
    Maximize \sum_k a_k*z_k
    S.t.     \sum_k z_k <= K
        Input:
            X: n*D data matrix
            Y: m*D data matrix
       Output:
            z: optimal selector (binary variable)
    """
    n, D = np.shape(X)
    m, _ = np.shape(Y)

    z_xx = inner_product_compute(x,x)
    z_yy = inner_product_compute(y,y)
    z_xy = inner_product_compute(x,y)
    z_xx = z_xx - np.sum(x**2, 0)
    z_yy = z_yy - np.sum(y**2, 0)

    a = 1/(n * (n-1)) * z_xx + 1/(m * (m-1)) * z_yy - 2/(n * m) * z_xy

    z = np.zeros(D)

    Indices = np.argsort(-a)
    z[Indices[:K]] = 1
    return z

def quad_product_compute(X, Y):
    """
    Compute sum_{i,j} (X_i[:]Y_j[:]) @ (X_i[:]Y_j[:]).T
        Input:
            X: n*D data matrix
            Y: m*D data matrix
       Output:
            A: D*D matrix
            a: D-dim array
    """
    n, D = np.shape(X)
    m, _ = np.shape(Y)
    A = np.zeros([D,D])
    a = np.zeros([D,1])
    for i in range(n):
        for j in range(m):
            a_xy = (X[i,:]*Y[j,:]).reshape([-1,1])
            A = A + a_xy @ a_xy.T
            a = a + a_xy
    return A, a.reshape([-1,])

def quad_kernel_selector(X, Y, sigma, z0, K=5):
    """
    Return optimal selector from quadratic kernel based on optimization
    Maximize zTAz + zTq
    S.t.     \sum_k z_k <= K
        Input:
            X: n*D data matrix
            Y: m*D data matrix
        sigma: kernel bandwidth
       Output:
            z: optimal selector (binary variable)
    """
    n, D = np.shape(X)
    m, _ = np.shape(Y)

    X2 = X**2
    Y2 = Y**2
    # formulate coeffficient
    A_xx, a_xx = quad_product_compute(x,x)
    A_xx = A_xx - X2.T @ X2
    a_xx = a_xx - np.sum(X2, 0)

    A_yy, a_yy = quad_product_compute(y,y)
    A_yy = A_yy - Y2.T @ Y2
    a_yy = a_yy - np.sum(Y2, 0)

    A_xy, a_xy = quad_product_compute(x,y)

    q = 1/(n * (n-1)) * a_xx + 1/(m * (m-1)) * a_yy - 2/(n * m) * a_xy
    q = q * 2 * sigma

    A = 1/(n * (n-1)) * A_xx + 1/(m * (m-1)) * A_yy - 2/(n * m) * A_xy
    A = -A
    model = gp.Model('qp')
    z = model.addVars(D, vtype=GRB.BINARY, name="z")
    z.start = z0
    #z = cp.Variable(D, boolean=True)
    #z = cp.Variable(D)
    model.setObjective(gp.quicksum(gp.quicksum(z[i] * z[j] * A[i,j] for i in range(D)) for j in range(D)) + gp.quicksum(q[i] * z[i] for i in range(D)))
    #objective = cp.Minimize(cp.quad_form(z, -A) + cp.sum(cp.multiply(q, z)))
    model.addConstr(z.sum() <= K)
    model.params.NonConvex = 2
    model.Params.LogToConsole = 0


    #constr = [cp.sum(z) <= K]
    model.optimize()

    list_sol = [v.X for v in z.values()]
    return np.array(list_sol)






# Parameter Setting
d_hist = [100]#[10, 20, 50, 100, 200]


for i in range(len(d_hist)):
    d = d_hist[i]
    N = 100

    x, y = utils.sample_generate(N,d)
    sigma = utils.kernelwidthPair(x,y)
    sigma = sigma/10

    a_xx,a_yy,a_xy = utils.MMD_exp_coeff_prepare(x, y, sigma)

    c = utils.kernelwidthPair(x,y)
    z0 = linear_kernel_selector(x, y, K=5)
    z_quad = quad_kernel_selector(x,y, np.sqrt(c)/100, z0, K = 5)

    #exp_kernel_selector(a_xx, a_yy, a_xy)
    #GRB_exp_kernel_selector_efficient(a_xx, a_yy, a_xy, z_quad)

    print(np.quantile(a_xx, 0.9))
    print(np.quantile(a_yy, 0.9))
    print(np.quantile(a_xy, 0.9))
    # print(np.min(a_xx))
    # print(np.max(a_yy))
    # print(len(np.where(a_xx == 1)[0]))
