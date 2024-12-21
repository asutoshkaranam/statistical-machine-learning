import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
feature_names = diabetes.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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

lambda_arr = np.logspace(np.log10(0.5), 4, num=50)

u_bar_arr = []
sigma_hat_arr = []

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for curr_lambda in lambda_arr:
    errors_per_fold_arr = []
    
    for train_ind, value_ind in kf.split(X_train):
        currfold_X_train, currfold_X_value = X_train[train_ind], X_train[value_ind]
        currfold_y_train, currfold_y_value = y_train[train_ind], y_train[value_ind]
        
        scaler = StandardScaler()
        X_train_fold_scaled = scaler.fit_transform(currfold_X_train)
        X_val_fold_scaled = scaler.transform(currfold_X_value)
        
        w_hat, b_hat = coordinate_descent(X_train_fold_scaled, currfold_y_train, lambdaa=curr_lambda)
        
        y_val_pred = np.dot(X_val_fold_scaled, w_hat) + b_hat
        
        errors_per_fold_arr.append(mean_squared_error(currfold_y_value, y_val_pred))
    
    u_bar = np.mean(errors_per_fold_arr)
    sigma_hat_u = np.std(errors_per_fold_arr) / np.sqrt(kf.get_n_splits())
    
    u_bar_arr.append(u_bar)
    sigma_hat_arr.append(sigma_hat_u)

# Plot the average validation error with standard error bars
plt.figure(figsize=(10, 6))
plt.errorbar(lambda_arr, u_bar_arr, yerr=sigma_hat_arr, fmt=':', capsize=3)
plt.xscale('log')
plt.xlabel('Log-Lambda')
plt.ylabel('Average Validation Error')
plt.title('Validation Error vs Lambda with Standard Error Bars')
plt.grid(True)
plt.tight_layout()
plt.show()

# Find the optimal lambda value
optimum_lambda_idx = np.argmin(u_bar_arr)
optimal_lambda = lambda_arr[optimum_lambda_idx]
print(f"Optimal lambda: {optimal_lambda}")