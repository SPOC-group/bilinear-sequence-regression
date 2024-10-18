import numpy as np
import cvxpy as cp

def min_norm(X, y, S):
    N, D, L = X.shape
    S_var = cp.Variable((D, L))

    objective = cp.Minimize(cp.norm(S_var, "nuc"))

    # The constraint is Tr[X * S] == y for each of the N samples
    constraints = [cp.sum(cp.multiply(X[n],S_var))/D/L == y[n] for n in range(N)]

    problem = cp.Problem(objective, constraints)

    # Low precision for faster convergence
    problem.solve(solver=cp.SCS)

    return np.mean((S_var.value - S)**2)


if __name__=="__main__":
    D = 50
    rho = 0.2
    beta = 0.5
    alpha = 0.2

    L = int(beta*D)
    M = int(rho*D)
    N = int(alpha*D*L) 

    X = np.random.randn(N, D, L)
    U = np.random.randn(D, M)
    V = np.random.randn(M, L)
    S = U @ V
    y = np.einsum('ndl,dl->n', X, S)/D/L


    overlap = min_norm(X, y, S) / M

    print(f"The overlap is {overlap}")