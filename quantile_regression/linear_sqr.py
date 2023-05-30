import numpy as np
from scipy.optimize import linprog
from scipy.sparse import block_diag
import matplotlib.pyplot as plt

np.set_printoptions(edgeitems=10, precision=2, linewidth=150)
np.random.seed(0)

class linear_sqr:
    
    def fit(self, X, y, taus):

        n = X.shape[0]
        k = X.shape[1]
        
        X_min = X.min()

        c_taus = []
        A_taus = []
        b_taus = []

        for tau in taus:

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
        A_ub = block_diag(A_taus)
        b_ub = np.hstack(b_taus)

        print("c:", c.shape)
        print(c)
        print("A_ub:", A_ub.shape)
        print(A_ub.toarray())
        print("b_ub:", b_ub.shape)

        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=(None, None))
        print(res['x'].shape)

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

    my_qr = linear_sqr()

    taus = [0.1,  0.9 ,0.5]
    my_qr.fit(X, y, taus)

    plt.scatter(X,y, label='sample')
    est_quantiles = my_qr.predict(X)
    ind = np.argsort(X[:,0])
    for i, tau in enumerate(sorted(taus)):
        plt.plot(X[ind,0], est_quantiles[ind,i], label="{:.0f}th Quantile".format(100*tau))
    plt.legend()
    plt.savefig("linear_sqr.png")
    plt.show()
