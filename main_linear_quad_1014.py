from xmlrpc.client import boolean
import numpy as np
import cvxpy as cp
from numpy.lib import utils
import torch
import scipy.spatial
import utils
import time
import gurobipy as gp
from gurobipy import GRB

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
    #return z.value




np.random.seed(1)

# Parameter Setting
d_hist = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
L_d_hist = len(d_hist)
max_trial = 100
counter_linear = np.zeros(L_d_hist)
counter_quadratic = np.zeros(L_d_hist)


for i in range(L_d_hist):
    d = d_hist[i]
    N = 100

    for trial in range(max_trial):

        x, y = utils.sample_generate(N,d)

        z0 = linear_kernel_selector(x, y, K=5)
        # print("----------------------------------------")
        # print("Dimension: ", d, "Sample Size: ",N)
        # print("Solution for Linear Kernel: ")
        # print(z0)

        c = utils.kernelwidthPair(x,y)
        z = quad_kernel_selector(x,y, np.sqrt(c)/100, z0, K = 5)
        # print("----------------------------------------")
        # print("Solution for Quadratic Kernel: ")
        # print(z)

        if z0[0] == 1:
            counter_linear[i] = counter_linear[i] + 1
        if z[0] == 1:
            counter_quadratic[i] = counter_quadratic[i] + 1
            print(z)
    print("----------------------------------------")
    print("Dimension: ", d, "Sample Size: ",N)
    print(counter_linear[i])
    print(counter_quadratic[i])

np.save("counter_linear_laplace.npy", counter_linear)
np.save("counter_quadratic_laplace.npy", counter_quadratic)


# # examine running time

# cpu_linear_hist = np.zeros(L_d_hist)
# cpu_quadratic_hist = np.zeros(L_d_hist)
# for i in range(L_d_hist):
#     d = d_hist[i]
#     N = d
#     x, y = utils.sample_generate_Gaussian(N,d)

#     start_time = time.time()
#     z0 = linear_kernel_selector(x, y, K=5)
#     cpu_linear = time.time() - start_time

#     start_time = time.time()
#     c = utils.kernelwidthPair(x,y)
#     z = quad_kernel_selector(x,y, np.sqrt(c)/100, z0, K = 5)
#     cpu_quadratic = time.time() - start_time

#     cpu_linear_hist[i] = cpu_linear
#     cpu_quadratic_hist[i] = cpu_quadratic

# np.save("cpu_linear_hist.npy", cpu_linear_hist)
# np.save("cpu_quadratic_hist.npy", cpu_quadratic_hist)




