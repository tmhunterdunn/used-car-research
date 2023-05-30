import numpy as np
from scipy.optimize import linprog
from scipy.sparse import block_diag
from quadprog import solve_qp
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel
import matplotlib.pyplot as plt

np.set_printoptions(edgeitems=10, precision=2, linewidth=150)
np.random.seed(0)





def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    qp_G = .5 * (P + P.T) + np.eye(P.shape[0])*0.00001  # make sure P is symmetric
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

class kernel_aqr:
    """Kernel quantile regression using the quantile as a feature."""

    def fit(self, X, y, num_taus, kernel):

        n = X.shape[0]
        k = X.shape[1]

        aug_data_lst = []

        taus = np.random.uniform(size=(n, num_taus))
        taus = np.sort(taus, axis=1)
        for i in range(num_taus):
            aug_data_i = np.hstack([X, taus[:, i:(i+1)]])
            aug_data_lst.append(aug_data_i)

        aug_X = np.vstack(aug_data_lst)
        aug_y = np.hstack(num_taus*[y])

        self.kernel = kernel
        L = kernel(aug_X, aug_X)
        self.design_mat = aug_X


        P = np.zeros((2*n*num_taus+1, 2*n*num_taus+1))
        P[n*num_taus+1:, n*num_taus+1:] = L
        c = np.ones((n*num_taus, 1))
        c = np.vstack([c, np.zeros((n*num_taus+1, 1))])

        A_top = np.diag(1/aug_X[:,-1])
        A_top = np.hstack([A_top, np.ones((n*num_taus,1)), L])
        A_top = -A_top

        A_bottom = np.diag(1 / (aug_X[:,-1]-1))
        A_bottom = np.hstack([A_bottom, np.ones((n*num_taus,1)), L])
        b_top = -aug_y
        b_bottom = aug_y

        A = np.vstack([A_top, A_bottom])
        b = np.hstack([b_top, b_bottom])



        weight_constr = np.zeros((n*(num_taus - 1), n*num_taus))

        for i in range(num_taus - 1):
            for j in range(n):
              weight_constr[i*n + j,:] = L[i*n + j, :]  -L[(i+1)*n + j, :]
              pass


        weight_constr = np.hstack([np.zeros((n*(num_taus-1), n*num_taus + 1)), weight_constr])
        q = c[:,0]
        G = np.vstack([A, weight_constr])
        h = np.hstack([b, np.zeros(n*(num_taus-1))])


        res = quadprog_solve_qp(P=P, q=q, G=G, h=h)
        print(res.shape)

        self.weights = np.expand_dims(res[n*num_taus:], 1)

    def predict(self, X, taus):

        exp_taus = np.expand_dims(taus, 1)
        aug_X = np.hstack([X, exp_taus])
        return self.kernel(aug_X, self.design_mat)  @ self.weights[1:,:] + self.weights[0,:]

if __name__ == '__main__':

    sample = np.random.multivariate_normal(mean=[0,0], cov = [[1, 0.75],
                                                              [0.75, 1]],
                                                       size = 200)

    X = np.expand_dims(sample[:,0], 1)
    y = sample[:,1]

    my_qr = kernel_aqr()

    num_tuas = 2
    my_qr.fit(X, y, 2, rbf_kernel)

    plt.scatter(X,y, label='sample')
    test_sample = np.random.multivariate_normal(mean=[0,0], cov = [[1, 0.75],
                                                              [0.75, 1]],
                                                       size = 100)

    test_X = np.expand_dims(test_sample[:,0], 1)
    taus = [0.2,0.3,0.5, 0.4,0.6, 0.7,0.8, 0.9]
    ind = np.argsort(test_X[:,0])

    for i, tau in enumerate(sorted(taus)):
        plt.plot(test_X[ind,0], my_qr.predict(test_X, np.repeat(tau,
            test_sample.shape[0]))[ind,0], label="{0:.0f}th Quantile".format(100*tau))

    plt.legend()
    plt.savefig("kernel_aqr.png")
    plt.show()
