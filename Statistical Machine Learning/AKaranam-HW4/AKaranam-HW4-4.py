import numpy as np
import matplotlib.pyplot as plt

n = 600
d = 1200
k = 120
variance = 1

def get_gaussian_data(n, d, k, variance):
    tru_w = np.zeros((d ,1))
    
    for j in range(k): tru_w[j] = (j+1) / k

    np.random.seed(42)
    X = np.random.normal(size = (n,d))
    gauss_data = np.random.normal(scale = np.sqrt(variance), size=(n,))

    y = np.reshape(np.dot(tru_w.T, X.T) + gauss_data.T, (n,))
    return (X, y)

def lambda_min(X, y):
    return 2*np.max(np.abs(np.dot(y.T - np.mean(y), X)))

def coordinate_descent(X, y, lambdaa, start_idx_w = None, delta = 0.001):
    n = X.shape[0]
    d = X.shape[1]
    w_previous_idx = np.ones((d,1))
    if start_idx_w is None:
        w = np.zeros((d,))
    else:
        w = start_idx_w

    c = np.zeros((d,))

    a = 2*np.sum(np.square(X), axis=0)

    while np.max(np.abs(w - w_previous_idx)) > delta:

        w_previous_idx = np.copy(w)

        wTran_XTran = np.dot(w.T, X.T)

        b = 1/n * np.sum(y - wTran_XTran)

        for k in range(d):
			#expand the ck term and make it easier to write.
            c[k] = 2*np.dot(X[:, k], y - (b + np.dot(w.T, X.T) - w[k]*X[:, k]))

            if c[k] < -1*lambdaa:
                w[k] = (c[k] + lambdaa) / a[k]
            elif c[k] > lambdaa:
                w[k] = (c[k] - lambdaa) / a[k]
            else:
                w[k] = 0

    return (w, b)

(X, y) = get_gaussian_data(n, d, k, variance)

lambda_max = lambda_min(X, y)
lambda_ratio = 1.5
delta = 0.001

current_lambda = lambda_max
lambda_vals = [lambda_max]
w_previous_idx = None

W = np.zeros((d, 1))

while (np.count_nonzero(W[:, -1]) != d):
    current_lambda = current_lambda / lambda_ratio
    print("Current lambda = ", current_lambda)

    (w_next, b) = coordinate_descent(X, y, start_idx_w=w_previous_idx, lambdaa=current_lambda, delta=delta)
    W = np.concatenate((W, np.expand_dims(w_next, axis=1)), axis=1)
    w_previous_idx = np.copy(w_next)
	
    lambda_vals.append(current_lambda)

plt.figure(1)
plt.semilogx(lambda_vals, np.count_nonzero(W, axis=0))
plt.xlabel('Log-Lambda')
plt.ylabel('Number of Nonzeroes in w')
plt.title('Number of Nonzero coefficients and Log-Lambda')
plt.show()

FDR = np.append([0], np.count_nonzero(W[k:, 1:], axis=0) / np.count_nonzero(W[:,1:], axis=0))

TPR = np.count_nonzero(W[:k, :], axis=0) / k

plt.figure(2)
plt.plot(FDR, TPR)
plt.title('False Discovery Rate values & True Positive Rate values')
plt.xlabel('FDR')
plt.ylabel('TPR')
plt.show()