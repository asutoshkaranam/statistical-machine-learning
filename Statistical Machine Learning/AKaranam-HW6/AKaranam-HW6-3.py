#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

dataset_size = 5000
plus_1_mu = np.array([-5, 5])
minus_1_mu = np.array([-2, 4])
covariance_mtrx = np.array([[2, 0], [0, 3]])
optimal_seperator_w = np.array([-1.5, 1/3])
optimal_seperator_b = -6.75

plus_1_dataset = np.random.multivariate_normal(plus_1_mu, covariance_mtrx, dataset_size)
minus_1_dataset = np.random.multivariate_normal(minus_1_mu, covariance_mtrx, dataset_size)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(plus_1_dataset[:, 0], plus_1_dataset[:, 1], c='blue', label='Y = 1', alpha=0.6)
ax.scatter(minus_1_dataset[:, 0], minus_1_dataset[:, 1], c='red', label='Y = -1', alpha=0.6)

x_axis_range = np.linspace(-10, 0, 100)
y_axis_range = (-optimal_seperator_w[0] * x_axis_range - optimal_seperator_b) / optimal_seperator_w[1]
ax.plot(x_axis_range, y_axis_range, label='Optimal Separator', color='green', linewidth=2)

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_title('Scatter Plot - Dataset and Optimal Separator')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()


# In[2]:


from scipy.special import expit

def sigmoid(x):
    return expit(x)

def get_features_normalized(X):
    mu = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)
    return (X - mu) / std_dev, mu, std_dev

def func_grad_descent(X, y, learn_rate=0.1, epochs=2000):
    feature_size = X.shape[1]
    w = np.random.randn(feature_size) / np.sqrt(feature_size)
    
    for epoch in range(epochs):
        z = np.dot(X, w)
        pred_results = sigmoid(z)
        delta_W = np.dot(X.T, (pred_results - (y + 1) / 2)) / len(y)
        w -= learn_rate * delta_W
        
        if epoch % 100 == 0 and epoch > 0:
            loss = -np.mean(y * z - np.log(1 + np.exp(z)))
            if loss < 1e-5:
                break
    
    return w


# In[3]:


X = np.vstack([plus_1_dataset, minus_1_dataset])
y = np.hstack([np.ones(dataset_size), -np.ones(dataset_size)])

X_normalized, X_mean, X_std = get_features_normalized(X)
X_normalized = np.hstack([X_normalized, np.ones((2*dataset_size, 1))])

log_reg_maxiter = 100
weights_matrix = np.zeros((log_reg_maxiter, 3))

for i in range(log_reg_maxiter):
    np.random.seed(i)
    weights_matrix[i] = func_grad_descent(X_normalized, y)

average_model_w = np.mean(weights_matrix, axis=0)

denorm_log_reg_w = np.zeros(3)
denorm_log_reg_w[:2] = average_model_w[:2] / X_std
denorm_log_reg_w[2] = average_model_w[2] - np.sum(average_model_w[:2] * X_mean / X_std)

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(plus_1_dataset[:, 0], plus_1_dataset[:, 1], c='blue', label='Y = 1', alpha=0.6)
ax.scatter(minus_1_dataset[:, 0], minus_1_dataset[:, 1], c='red', label='Y = -1', alpha=0.6)

x_axis_range = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)

y_range_opt = (-optimal_seperator_w[0] * x_axis_range - optimal_seperator_b) / optimal_seperator_w[1]
ax.plot(x_axis_range, y_range_opt, 'g-', label='Optimal Separator', linewidth=2)

y_range_log = (-denorm_log_reg_w[0] * x_axis_range - denorm_log_reg_w[2]) / denorm_log_reg_w[1]
ax.plot(x_axis_range, y_range_log, 'y--', label='Logistic Regression', linewidth=2)

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_title('Comparison of Optimal and Logistic Regression Boundaries')
ax.legend()
ax.grid(True)

optimal_separator_eqn = f'Optimal: {optimal_seperator_w[0]:.2f}x₁ + {optimal_seperator_w[1]:.2f}x₂ + {optimal_seperator_b:.2f} = 0'
log_reg_eqn = f'Logistic: {denorm_log_reg_w[0]:.2f}x₁ + {denorm_log_reg_w[1]:.2f}x₂ + {denorm_log_reg_w[2]:.2f} = 0'
plt.text(0.05, 0.95, optimal_separator_eqn, transform=ax.transAxes, 
         bbox=dict(facecolor='white', alpha=0.8))
plt.text(0.05, 0.88, log_reg_eqn, transform=ax.transAxes,
         bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()


# In[4]:


def compute_sigmoid_scores(X, weights):
    return expit(np.dot(X, weights))

def compute_tpr_fpr(y_true, final_score, threshold):
    pred_results = (final_score >= threshold).astype(int)
    y_true = (y_true + 1) / 2
    
    true_positives = np.sum((pred_results == 1) & (y_true == 1))
    false_positives = np.sum((pred_results == 1) & (y_true == 0))
    
    total_positives = np.sum(y_true == 1)
    total_negatives = np.sum(y_true == 0)
    
    tpr = true_positives / total_positives
    fpr = false_positives / total_negatives
    
    return tpr, fpr

X_with_bias = np.hstack([X, np.ones((2*dataset_size, 1))])

final_score = compute_sigmoid_scores(X_with_bias, denorm_log_reg_w)

max_trshold = np.linspace(0, 1, 50)

tpr_arr = []
fpr_arr = []

for threshold in max_trshold:
    tpr, fpr = compute_tpr_fpr(y, final_score, threshold)
    tpr_arr.append(tpr)
    fpr_arr.append(fpr)

tpr_arr = np.array(tpr_arr)
fpr_arr = np.array(fpr_arr)

sort_idx = np.argsort(fpr_arr)
fpr_arr = fpr_arr[sort_idx]
tpr_arr = tpr_arr[sort_idx]

plt.figure(figsize=(8, 6))
plt.plot(fpr_arr, tpr_arr, 'b-', linewidth=2, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Plot - Logistic Regression')

plt.grid(True)
plt.legend()
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.tight_layout()

plt.show()


# In[6]:


from sklearn.preprocessing import StandardScaler

X = np.vstack((plus_1_dataset, minus_1_dataset))
y = np.hstack((np.ones(dataset_size), -np.ones(dataset_size)))

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

X_normalized_with_bias = np.hstack((X_normalized, np.ones((2*dataset_size, 1))))

w_bar = np.zeros(3)
n_iter = 0
misclassified = True

while misclassified:
    if n_iter > 5000:
        break
    misclassified = False
    for i in range(2 * dataset_size):
        xi = X_normalized_with_bias[i, :]
        yi = y[i]
        if yi * np.dot(w_bar, xi) <= 0:
            w_bar = w_bar + yi * xi
            misclassified = True
    n_iter += 1
    if not misclassified:
        break

plt.figure(figsize=(10, 8))
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label='Y => 1')
plt.scatter(X[y == -1, 0], X[y == -1, 1], c='red', label='Y => -1')

x_data_points = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
norm_x_data = (x_data_points - scaler.mean_[0]) / scaler.scale_[0]
norm_y_data = (-w_bar[0] * norm_x_data - w_bar[2]) / w_bar[1]
denorm_y = norm_y_data * scaler.scale_[1] + scaler.mean_[1]

plt.plot(x_data_points, denorm_y, label='Averaged Model (w-bar)', color='orange')

new_optimal_seperator_w = optimal_seperator_w
new_optimal_seperator_b = -6.75

optimal_separator_y = (-new_optimal_seperator_w[0] * x_data_points - new_optimal_seperator_b) / new_optimal_seperator_w[1]

plt.plot(x_data_points, optimal_separator_y, label='Optimal Separator w*', color='green')

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Scatter Plot - Dataset & Averaged Perceptron Model vs Optimal Separator')
plt.legend()
plt.grid(True)
plt.show()


# In[5]:


from sklearn.preprocessing import StandardScaler

cov_matrix_adjusted = covariance_mtrx / 10

X_pos = np.random.multivariate_normal(plus_1_mu, cov_matrix_adjusted, dataset_size)
X_neg = np.random.multivariate_normal(minus_1_mu, cov_matrix_adjusted, dataset_size)

new_optimal_seperator_w = np.linalg.inv(cov_matrix_adjusted) @ (plus_1_mu - minus_1_mu)
new_optimal_seperator_b = -0.5 * (plus_1_mu.T @ np.linalg.inv(cov_matrix_adjusted) @ plus_1_mu -
                       minus_1_mu.T @ np.linalg.inv(cov_matrix_adjusted) @ minus_1_mu)

X = np.vstack((X_pos, X_neg))
y = np.hstack((np.ones(dataset_size), -np.ones(dataset_size)))

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

X_normalized_with_bias = np.hstack((X_normalized, new_optimal_seperator_b * np.ones((2*dataset_size, 1))))

w_bar = np.zeros(3)
n_iter = 0
misclassified = True

while misclassified:
    if n_iter > 5000:
        break
    misclassified = False
    for i in range(2 * dataset_size):
        xi = X_normalized_with_bias[i, :]
        yi = y[i]
        if yi * np.dot(w_bar, xi) <= 0:
            w_bar = w_bar + yi * xi
            misclassified = True
    n_iter += 1
    if not misclassified:
        break

plt.figure(figsize=(10, 8))
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label='Y = 1')
plt.scatter(X[y == -1, 0], X[y == -1, 1], c='red', label='Y = -1')

x_data_points = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
norm_x_data = (x_data_points - scaler.mean_[0]) / scaler.scale_[0]
norm_y_data = (-w_bar[0] * norm_x_data - w_bar[2]) / w_bar[1]
denorm_y = norm_y_data * scaler.scale_[1] + scaler.mean_[1]

plt.plot(x_data_points, denorm_y, label='Averaged Model w-bar', color='orange')

optimal_separator_y = (-new_optimal_seperator_w[0] * x_data_points - new_optimal_seperator_b) / new_optimal_seperator_w[1]

plt.plot(x_data_points, optimal_separator_y, label='Optimal Separator w*', color='green')

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Scatter Plot - Dataset & Averaged Model & Optimal Separator')
plt.legend()
plt.grid(True)
plt.show()

print("Number of iterations for the new setting:", n_iter)


# In[ ]:




