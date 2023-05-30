import numpy as np
from scipy.optimize import linprog
from scipy.sparse import block_diag
from quadprog import solve_qp
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
import matplotlib.pyplot as plt

np.set_printoptions(edgeitems=10, precision=2, linewidth=150)
np.random.seed(0)


def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    penalty_lambda = 0.5
    qp_G = penalty_lambda * (P + P.T) + np.eye(P.shape[0])*0.00001  # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -numpy.vstack([A, G]).T
        qp_b = -numpy.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

class kernel_snqr:

    def fit(self, X, y, taus, kernel):

        n = X.shape[0]
        k = X.shape[1]

        self.kernel = kernel
        L = kernel(X,X)
        self.design_mat = X

        self.L_X = min(0, np.min(L))
        L_plus = L - self.L_X
        print(np.linalg.eigvals(L_plus))
        print(np.linalg.eigvals(L))

        sort_taus = sorted(taus)

        P_taus = []
        c_taus = []
        A_taus = []
        b_taus = []

        for tau in sort_taus:

            P_tau = np.zeros((2*n+1, 2*n+1))
            P_tau[n+1:, n+1:] = L_plus
            P_taus.append(P_tau)
            c_tau = np.ones((n, 1))
            c_tau = np.vstack([c_tau, np.zeros((n+1, 1))])
            c_taus.append(c_tau)

            A_top = np.eye(n)/tau
            A_top = np.hstack([A_top, np.ones((n,1)), L_plus])
            A_top = -A_top
            A_bottom = np.eye(n)/(tau-1)
            A_bottom = np.hstack([A_bottom, np.ones((n,1)), L_plus])
            b_top = -y
            b_bottom = y

            A_tau = np.vstack([A_top, A_bottom])
            b_tau = np.hstack([b_top, b_bottom])

            print("A_tau:", A_tau.shape)

            A_taus.append(A_tau)
            b_taus.append(b_tau)

        P = block_diag(P_taus)
        c = np.vstack(c_taus)
        A_ub = block_diag(A_taus).toarray()
        b_ub = np.hstack(b_taus)

        weight_constr = np.zeros(((n+1)*(len(taus)-1), A_ub.shape[1]))
        for i in range(len(taus)-1):
            inds_tau1 = list(range(n+(2*n+1)*i, (2*n+1)*i+2*n+1))
            inds_tau2 = list(range(n+(2*n+1)*(i+1), (2*n+1)*(i+1)+2*n+1))
            for j in range(len(inds_tau1)):
                weight_constr[i+j, inds_tau1[j]] = 1
                weight_constr[i+j, inds_tau2[j]] = -1




        P = P.toarray()
        q = c[:,0]
        G = np.vstack([A_ub, weight_constr])
        h = np.hstack([b_ub, np.zeros((n+1)*(len(taus)-1))])

        print("c:", c.shape)
        print("A_ub:", A_ub.shape)
        print("b_ub:", b_ub.shape)

        res = quadprog_solve_qp(P=P, q=q, G=G, h=h)
        print(res.shape)

        self.beta_ks = np.vstack([res[n+(2*n+1)*i:(2*n+1)*i+2*n+1] for i in range(len(taus))]).T
        print("beta_ks:", self.beta_ks.shape)

    def predict(self, X):

        return (self.kernel(X, self.design_mat) - self.L_X) @ self.beta_ks[1:,:] + self.beta_ks[0,:]

if __name__ == '__main__':

    sample = np.random.multivariate_normal(mean=[0,0], cov = [[1, 0.75],
                                                              [0.75, 1]],
                                                       size = 100)

    X = np.expand_dims(sample[:,0], 1)
    y = sample[:,1]

    my_qr = kernel_snqr()

    taus = [ 0.9, 0.8,0.4 ,0.5 ,0.1 ]
    my_qr.fit(X, y, taus, rbf_kernel)

    plt.scatter(X,y, label='sample')
    est_quantiles = my_qr.predict(X)
    ind = np.argsort(X[:,0])

    for i, tau in enumerate(sorted(taus)):
        plt.plot(X[ind,0], est_quantiles[ind,i], label="{:.0f}th Quantile".format(100*tau))

    plt.legend()
    plt.savefig("kernel_snqr.png")
    plt.show()
