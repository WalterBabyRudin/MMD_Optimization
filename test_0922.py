import cvxpy as cp
import numpy as np

np.random.seed(0)
m, n= 40, 25

A = np.random.rand(m, n)
b = np.random.randn(n)


Q = A.T @ A - 0.4 * np.eye(n)
x = cp.Variable(n, integer=True)
objective = cp.Minimize(cp.quad_form(x, Q) + cp.sum(cp.multiply(b, x)))
prob = cp.Problem(objective)
prob.solve(solver = cp.GUROBI)

print(x.value)
print(cp.installed_solvers())