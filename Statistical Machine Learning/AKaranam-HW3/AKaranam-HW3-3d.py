import numpy as np
import matplotlib.pyplot as plt

def get_matrix(d):
    A = np.random.uniform(-1, 1, (d, d))
    A += np.eye(d) * 0.01
    sigma_matrix = np.dot(A.T, A)
    return sigma_matrix

trail_count = 30
given_lambda = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

train_error_arr = np.zeros((len(given_lambda), trail_count))
test_error_arr = np.zeros((len(given_lambda), trail_count))

rng = np.random.default_rng()
d = 100
mu = rng.standard_normal(size=d)
Cov = get_matrix(d)
sigma_noise = rng.uniform(0.3, 0.7)
a_true = rng.normal(0, 1, size=(d, 1))

for j in range(trail_count):
    train_n = 100
    test_n = 1000
    
    # Generate training and test data using fixed parameters
    X_train = rng.multivariate_normal(mu, Cov, size=train_n)
    y_train = X_train.dot(a_true) + rng.normal(0, sigma_noise, size=(train_n, 1))
    X_test = rng.multivariate_normal(mu, Cov, size=test_n)
    y_test = X_test.dot(a_true) + rng.normal(0, sigma_noise, size=(test_n, 1))

    for i, curr_lambda in enumerate(given_lambda):
        term_1 = np.dot(X_train.T, X_train) + curr_lambda * np.eye(d)
        term_2 = np.dot(X_train.T, y_train)
        wcap = np.linalg.solve(term_1, term_2)
        test_err = np.linalg.norm(np.dot(X_test, wcap) - y_test) / np.linalg.norm(y_test)
        train_err = np.linalg.norm(np.dot(X_train, wcap) - y_train) / np.linalg.norm(y_train)
        test_error_arr[i, j] = test_err
        train_error_arr[i, j] = train_err

# Calculate the average along each row (each lambda value)
ave_train_errors = np.mean(train_error_arr, axis=1)
ave_test_errors = np.mean(test_error_arr, axis=1)

plt.figure(figsize=(12, 8))
plt.semilogx(given_lambda, ave_train_errors, 'b-o', label='Training error')
plt.semilogx(given_lambda, ave_test_errors, 'r-o', label='Testing error')
plt.legend()
plt.ylabel('Normalized Error')
plt.xlabel('lambda')
plt.title('Ridge Regression: Training & Testing Errors vs Lambda')
plt.grid(True)
plt.show()
