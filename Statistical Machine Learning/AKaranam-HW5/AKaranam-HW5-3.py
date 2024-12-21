import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

def get_gradient_j(X, y, w, b, given_lambda):
    n = len(y)
    
    y = y.reshape(-1, 1)
    z = b + X @ w
    mu = 1 / (1 + np.exp(-y * z))
    
    eqn_w = (1/n * X.T @ ((mu - 1) * y)) + 2 * given_lambda * w
    eqn_b = np.mean((mu - 1) * y)
    
    return np.vstack((eqn_w, eqn_b))

def compute_J(X, y, w,b, given_lambda):
    inner_term = np.multiply(-y, b + np.dot(w.T, X.T))
    n = y.size

    return np.squeeze(1/n * np.sum(np.log(1 + np.exp(inner_term))) + given_lambda * np.dot(w.T, w))

def compute_misclassification_error(X, y, w, b):
    n = y.size
    results = np.sign(b + np.dot(w.T, X.T))

    results = np.where(results == 0, 1, results)

    err = 1 - np.sum( np.abs(results + y) ) / (2*n)
    return err


def gradient_descent(init_vector_x, gradient_compute_fxn, best_rate=0.1, delta=1e-4, max_iterations=1000):
    x = np.array(init_vector_x, dtype=np.float64)
    x_arr_full = [x.copy()]
    
    for _ in range(max_iterations):
        grad = gradient_compute_fxn(x)
        if np.max(np.abs(grad)) <= delta:
            break
        
        x -= best_rate * grad
        x_arr_full.append(x.copy())
    
    return x, x_arr_full


def stochastic_gradient_descent_func(X, y, init_vector_x, gradient_compute_fxn, size_of_the_batch, best_rate=0.1, iteration_num=100):
    x = init_vector_x
    n = y.size
    x_arr_full = [x]

    itr = 0
    while itr < iteration_num:
        train_list_order = list(range(n))
        random.shuffle(train_list_order)

        for itr_batch in range(n // size_of_the_batch):

            itr += 1
            curr_sample = train_list_order[itr_batch * size_of_the_batch : (itr_batch + 1) * size_of_the_batch]
            X_batch = X[curr_sample, :]
            y_batch = y[curr_sample]

            grad = gradient_compute_fxn(X_batch=X_batch, y_batch=y_batch, x=x)
            x = x - best_rate * grad

            x_arr_full.append(x)
            if itr > iteration_num: break


        else:
            if n % size_of_the_batch != 0:
                itr += 1
                curr_sample = train_list_order[(n // size_of_the_batch) * size_of_the_batch : ]
                X_batch = X[curr_sample ,:]
                y_batch = y[curr_sample]

                grad = gradient_compute_fxn(X_batch, y_batch, x)
                x = x - best_rate * grad

                x_arr_full.append(x)

    return (x, x_arr_full)

# -----------------------------------------------------------------------------------------------------------------------------------

mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist.data, mnist.target.astype(int)

# Split the dataset into train and test sets - the original dataset has 60000 samples of train and 10000 samples of test. Hence, hard-coding
X_train, X_test = X[:60000], X[60000:]
labels_train, labels_test = y[:60000], y[60000:]

# Normalize the data by a factor
X_train = X_train / 255.0
X_test = X_test / 255.0

print("Filtering the dataset to extract 9s & 6s")
train_mask = (labels_train == 9) | (labels_train == 6)
test_mask = (labels_test == 9) | (labels_test == 6)

# Extract the entries matching this pattern
X_train = X_train[train_mask]
X_test = X_test[test_mask]

# Convert label for a 6 to 1 and a 9 to -1
Y_train = labels_train[train_mask]
Y_train = np.where(Y_train == 6, 1, -1)

Y_test = labels_test[test_mask]
Y_test = np.where(Y_test == 6, 1, -1)

print(f"Training set shape: {X_train.shape}")
print(f"Training labels shape: {Y_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Test labels shape: {Y_test.shape}")
print("Training set:")
print(f"Class -1 (digit 9): {np.sum(Y_train == -1)}")
print(f"Class 1 (digit 6): {np.sum(Y_train == 1)}")
print("Test set:")
print(f"Class -1 (digit 9): {np.sum(Y_test == -1)}")
print(f"Class 1 (digit 6): {np.sum(Y_test == 1)}")

# -------------------------------------------------------------------------------------------------------------------------------------

given_lambda = 0.1

# Start with defining the initial vector & gradient func.
init_vector_x = np.zeros((X_train.shape[1]+1, 1))
training_grad_func = lambda x: get_gradient_j(X_train, Y_train, x[:-1], x[-1], given_lambda)
delta = 0.01
best_rate = 0.1 # fastest rate for convergence

(x_best_train, training_x_arr_full) = gradient_descent(init_vector_x, training_grad_func, best_rate, delta)


# 3-c i

plt.figure(1)
plt.plot([compute_J(X=X_train, y=Y_train, w=x[:-1], b=x[-1], given_lambda=given_lambda) for x in training_x_arr_full])
plt.plot([compute_J(X=X_test, y=Y_test, w=x[:-1], b=x[-1], given_lambda=given_lambda) for x in training_x_arr_full])
plt.title('J(w,b) function vs iterations')
plt.ylabel('Function J(w,b)')
plt.xlabel('Iteration #')
plt.legend(['Training', 'Testing'])
plt.show()

# 3-c ii

plt.figure(2)
plt.plot([compute_misclassification_error(X=X_train, y=Y_train, w=x[:-1], b=x[-1]) for x in training_x_arr_full[1:]])
plt.plot([compute_misclassification_error(X=X_test, y=Y_test, w=x[:-1], b=x[-1]) for x in training_x_arr_full[1:]])
plt.title('Misclassification Error vs iterations')
plt.ylabel('Misclassification Errors')
plt.xlabel('Iteration #')
plt.legend(['Training', 'Testing'])
plt.show()

# --------------------------------------------------------------------------------------------------------------------------------------------

# 3-d
init_vector_x = np.zeros((X_train.shape[1]+1, 1))
size_of_the_batch = 1
best_rate = 0.01
num_iterations = 200

#Start taking batches
stochastic_gradient_descent = lambda X_batch, y_batch, x: get_gradient_j(X_batch, y_batch, x[:-1], x[-1], given_lambda)
(stochastic_gd_best_x, stochastic_gd_x_arr_batch_1) = stochastic_gradient_descent_func(X=X_train, y=Y_train, init_vector_x=init_vector_x, gradient_compute_fxn=stochastic_gradient_descent, \
                               size_of_the_batch=size_of_the_batch, best_rate=best_rate, iteration_num=num_iterations)

# 3-d i

plt.figure(3)
plt.plot([compute_J(X=X_train, y=Y_train, w=x[:-1], b=x[-1], given_lambda=given_lambda) for x in stochastic_gd_x_arr_batch_1])
plt.plot([compute_J(X=X_test, y=Y_test, w=x[:-1], b=x[-1], given_lambda=given_lambda) for x in stochastic_gd_x_arr_batch_1])
plt.title('Stochastic Gradient Descent Function vs iterations with batch size 1')
plt.ylabel('Function J(w,b)')
plt.xlabel('Iteration #')
plt.legend(['Training', 'Testing'])
plt.show()


# 3-d ii

plt.figure(4)
plt.plot([compute_misclassification_error(X=X_train, y=Y_train, w=x[:-1], b=x[-1]) for x in stochastic_gd_x_arr_batch_1[1:]])
plt.plot([compute_misclassification_error(X=X_test, y=Y_test, w=x[:-1], b=x[-1]) for x in stochastic_gd_x_arr_batch_1[1:]])
plt.title('Stochastic Gradient Descent Misclassification Errors vs iterations with batch size 1')
plt.ylabel(' Misclassification Errors')
plt.xlabel('Iteration #')
plt.legend(['Training', 'Testing'])
plt.show()


# -------------------------------------------------------------------------------------------------------------------------------------------------

# 3-e
n = X_train.shape[0]
d = X_train.shape[1]
init_vector_x = np.zeros((d+1, 1))
size_of_the_batch = 100
best_rate = 0.1
given_lambda = 0.1

num_iterations = 150


stochastic_gradient_descent = lambda X_batch, y_batch, x: get_gradient_j(X_batch, y_batch, x[:-1], x[-1], given_lambda)
(stochastic_gd_best_x, stochastic_gd_x_arr_batch_100) = stochastic_gradient_descent_func(X=X_train, y=Y_train, init_vector_x=init_vector_x, gradient_compute_fxn=stochastic_gradient_descent, \
                                  size_of_the_batch=size_of_the_batch, best_rate=best_rate, iteration_num=num_iterations)

# 3-e i

plt.figure(5)
plt.plot([compute_J(X=X_train, y=Y_train, w=x[:-1], b=x[-1], given_lambda=given_lambda) for x in stochastic_gd_x_arr_batch_100])
plt.plot([compute_J(X=X_test, y=Y_test, w=x[:-1], b=x[-1], given_lambda=given_lambda) for x in stochastic_gd_x_arr_batch_100])
plt.title('Stochastic Gradient Descent Function vs iterations with batch size 100')
plt.ylabel('J(w,b)')
plt.xlabel('Iteration Number')
plt.legend(['Training', 'Testing'])
plt.show()


# 3-e ii

plt.figure(6)
plt.plot([compute_misclassification_error(X=X_train, y=Y_train, w=x[:-1], b=x[-1]) for x in stochastic_gd_x_arr_batch_100[1:]])
plt.plot([compute_misclassification_error(X=X_test, y=Y_test, w=x[:-1], b=x[-1]) for x in stochastic_gd_x_arr_batch_100[1:]])
plt.title('Stochastic Gradient Descent Misclassification Errors vs iterations with batch size 100')
plt.ylabel('Error rate')
plt.xlabel('Iteration Number')
plt.legend(['Training', 'Testing'])
plt.show()
