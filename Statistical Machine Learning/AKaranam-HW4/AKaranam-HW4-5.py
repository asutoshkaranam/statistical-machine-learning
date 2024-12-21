import numpy as np
import matplotlib.pyplot as plt

def get_gaussian_data(n, d, k, variance, seed):
    tru_w = np.zeros((d ,1))

    for j in range(k): tru_w[j] = (j+1) / k

    np.random.seed(seed)
    X = np.random.normal(size = (n,d))
    gauss_data = np.random.normal(scale = np.sqrt(variance), size=(n,))

    y = np.reshape(np.dot(tru_w.T, X.T) + gauss_data.T, (n,))
    return (X, y)

def ridge_regression(X, y, lam):
    d = X.shape[1]
    Xt_X_plus_lambda = np.dot(X.T, X) + lam * np.eye(d)
    Xt_y = np.dot(X.T, y)
    w_hat = np.linalg.solve(Xt_X_plus_lambda, Xt_y)
    return w_hat

# Generate data using the previous set of values
n_train = 600
n_test = 600
d = 1200
k = 120
variance = 1

#Assume Same underlying distribution with different noise factor, so change the seed
(X_train, y_train) = get_gaussian_data(n_train, d, k, variance, seed=123)
(X_test, y_test) = get_gaussian_data(n_test, d, k, variance, seed=456)

#Start Lambda max from 1000
lambda_max = 100000
lambda_ratio = 1.5
lambda_arr = []

non_zero_coeffs_arr = []
l2_norm_coeffs_arr = []

curr_lambda = lambda_max
while curr_lambda > 0.001:
    w_hat = ridge_regression(X_train, y_train, curr_lambda)
    non_zero_count = np.count_nonzero(np.abs(w_hat) > 1e-5) #make way for slight deviations from exact zero and floating points
    l2_norm = np.linalg.norm(w_hat)
    non_zero_coeffs_arr.append(non_zero_count)
    l2_norm_coeffs_arr.append(l2_norm)
    lambda_arr.append(curr_lambda)
    curr_lambda /= lambda_ratio

plt.subplot(2, 1, 1)
plt.semilogx(lambda_arr, non_zero_coeffs_arr, 'r-')
plt.xlabel('Log-Lambda')
plt.ylabel('Nonzero Coefficients')
plt.title('Number of Nonzero Coefficients vs Log-Lambda in Ridge Solution')
plt.ticklabel_format(style='plain', axis='y')
plt.gca().get_yaxis().get_major_formatter().set_scientific(False)

plt.subplot(2, 1, 2)
plt.semilogx(lambda_arr, l2_norm_coeffs_arr, 'b-')
plt.xlabel('Log-Lambda')
plt.ylabel('L2 Norm of w')
plt.title('L2 Norm vs Log-Lambda of Ridge Solution')

plt.tight_layout()
plt.show()
