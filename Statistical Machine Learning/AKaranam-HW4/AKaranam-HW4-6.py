import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
feature_names = diabetes.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training set shape:", X_train_scaled.shape)
print("Test set shape:", X_test_scaled.shape)

#Verify the Scaler
print("Training data mean:", X_train_scaled.mean(axis=0))
print("Training data variance:", X_train_scaled.var(axis=0))

def lambda_min(X, y):
    return 2*np.max(np.abs(np.dot(y.T - np.mean(y), X)))

def coordinate_descent(X, y, lambdaa, start_idx_w = None, delta = 0.0001):
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
	
lambda_max = lambda_min(X_train_scaled, y_train)
lambda_ratio = 2 

curr_lambda = lambda_max
lambda_arr = []
delta = 0.0001
w_previous_idx = None

W = None
B = []

train_errors = []
test_errors = []

while curr_lambda >= 0.01:

    lambda_arr.append(curr_lambda)

    (w, b) = coordinate_descent(X_train_scaled, y_train, lambdaa=curr_lambda, start_idx_w=w_previous_idx, delta=delta)

    if W is None:
        W = np.expand_dims(w, axis=1)
    else:
        W = np.concatenate((W, np.expand_dims(w, axis=1)), axis=1)

    B.append(b)
	
    y_train_pred = np.dot(X_train_scaled, w) + b
    y_test_pred = np.dot(X_test_scaled, w) + b
    train_errors.append(mean_squared_error(y_train, y_train_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))
	
    w_previous_idx = np.copy(w)
    curr_lambda /= lambda_ratio

plt.figure(1)
plt.semilogx(lambda_arr, np.count_nonzero(W, axis=0))
plt.xlabel('Log-Lambda')
plt.ylabel('Number of Nonzeroes in w')
plt.title('Number of Nonzero coefficients and Log-Lambda')
plt.show()

# Part b
# Plot regularization paths
plt.figure(2)
for i in range(W.shape[0]):
    plt.plot(lambda_arr, W[i, :], label=feature_names[i])

plt.xscale('log')
plt.xlabel('Log-Lambda')
plt.ylabel('Weights')
plt.title('Regularization Paths for Lasso')
plt.legend()
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

#Part C
plt.figure(3)
plt.semilogx(lambda_arr, train_errors, label='Train Error')
plt.semilogx(lambda_arr, test_errors, label='Test Error')
plt.xlabel('Log-Lambda')
plt.ylabel('Mean Squared Error')
plt.title('Train & Test Error vs Log-Lambda')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate LASSO weights for lambda = 4000
lambda_fixed = 4000
w, b = coordinate_descent(X_train_scaled, y_train, lambdaa=lambda_fixed)

# Print feature weights
print(f"LASSO coefficients for fixed lambda = {lambda_fixed}:")
for name, weight in zip(feature_names, w):
    print(f"{name}: {weight:.6f}")

# Find features with largest positive and negative weights
arg_max = np.argmax(w)
arg_min = np.argmin(w)

print(f"\n The Feature with the largest positive weight: {feature_names[arg_max]} ({w[arg_max]:.6f})")
print(f"The Feature with the largest negative weight: {feature_names[arg_min]} ({w[arg_min]:.6f})")
