import numpy as np
from scipy.optimize import linprog
from scipy.sparse import block_diag
import matplotlib.pyplot as plt

np.set_printoptions(edgeitems=10, precision=2, linewidth=150)
np.random.seed(0)

class linear_snqr:

    def fit(self, X, y, taus):

        n = X.shape[0]
        k = X.shape[1]

        X_min = np.expand_dims(X.min(axis=0), 1)
        X_max = np.expand_dims(X.max(axis=0), 1)
        sort_taus = sorted(taus)

        c_taus = []
        A_taus = []
        b_taus = []

        for tau in sort_taus:

            c_tau = np.ones((n, 1))
            c_tau = np.vstack([c_tau, np.zeros((k+1, 1))])
            c_taus.append(c_tau)

            A_top = np.eye(n)/tau
            A_top = np.hstack([A_top, np.ones((n,1)), X])
            A_top = -A_top
            A_bottom = np.eye(n)/(tau-1)
            A_bottom = np.hstack([A_bottom, np.ones((n,1)), X])
            b_top = -y
            b_bottom = y

            A_tau = np.vstack([A_top, A_bottom])
            b_tau = np.hstack([b_top, b_bottom])

            print("A_tau:", A_tau.shape)

            A_taus.append(A_tau)
            b_taus.append(b_tau)







        c = np.vstack(c_taus)
        A_ub = block_diag(A_taus).toarray()
        b_ub = np.hstack(b_taus)

        non_cross_A_min = np.zeros((len(taus)-1, A_ub.shape[1]))

        for i in range(len(sort_taus)-1):
            non_cross_A_min[i, n+(n+k+1)*i:n+(n+k+1)*i+k+1] = np.hstack([np.ones((1,1)), X_min.T])
            non_cross_A_min[i, n+(n+k+1)*(i+1):n+(n+k+1)*(i+1)+k+1] = -np.hstack([np.ones((1,1)), X_min.T])


        non_cross_A_max = np.zeros((len(taus)-1, A_ub.shape[1]))

        for i in range(len(sort_taus)-1):
            non_cross_A_max[i, n+(n+k+1)*i:n+(n+k+1)*i+k+1] = np.hstack([np.ones((1,1)), X_max.T])
            non_cross_A_max[i, n+(n+k+1)*(i+1):n+(n+k+1)*(i+1)+k+1] = -np.hstack([np.ones((1,1)), X_max.T])


        A_ub = np.vstack([A_ub, non_cross_A_min, non_cross_A_max])
        b_ub = np.hstack([b_ub, np.zeros(2*(len(taus)-1))])





        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=(None, None))

        self.beta_ks = np.vstack([res['x'][n+(n+k+1)*i:n+(n+k+1)*i+k+1] for i in range(len(taus))]).T
        print([(n+(n+k+1)*i, n+(n+k+1)*i+k+1) for i in range(len(taus))])
        print([res['x'][n+(n+k+1)*i:n+(n+k+1)*i+k+1] for i in range(len(taus))])
        print("beta_ks:", self.beta_ks.shape)

    def predict(self, X):

        return X @ self.beta_ks[1:,:] + self.beta_ks[0,:]

if __name__ == '__main__':

    sample = np.random.multivariate_normal(mean=[0,0], cov = [[1, 0.75],
                                                              [0.75, 1]],
                                                       size = 100)

    X = np.expand_dims(sample[:,0], 1)
    y = sample[:,1]

    my_qr = linear_snqr()

    taus = [0.1,  0.9 ,0.5]
    my_qr.fit(X, y, taus)

    plt.scatter(X,y, label='sample')
    est_quantiles = my_qr.predict(X)
    ind = np.argsort(X[:,0])
    for i, tau in enumerate(sorted(taus)):
        plt.plot(X[ind,0], est_quantiles[ind,i], label="{:.0f}th Quantile".format(100*tau))
    plt.legend()
    plt.savefig("linear_snqr.png")
    plt.show()
