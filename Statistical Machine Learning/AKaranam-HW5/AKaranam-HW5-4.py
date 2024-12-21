#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torchvision
from scipy.special import softmax
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

#code-up the one-hot encoder logic
def encoder_fxn_one_hot(labels, multiclass_size=10):
    return np.eye(multiclass_size)[labels]

#Code-up the train() function - implement an SGD algorithm as earlier and extend it to support multiple classes
def train(X, Y, regularization_lambdaa, best_learn_rate=0.01, epoch_size=50, batch=256):
    n, d = X.shape
    k = Y.shape[1]
    W = np.random.randn(d, k) * 0.01 #initialize the weights with random data to avoid errors
    
    #build a progressive un-shuffled mini-batch SGD for multiclass
    for epoch in range(epoch_size):
        for i in range(0, n, batch):
            curr_batch_of_X = X[i:i+batch]
            curr_batch_of_Y = Y[i:i+batch]
            
            # Forward pass - batch wise
            Z_Product_XW = curr_batch_of_X @ W
            classification_res = softmax(Z_Product_XW, axis=1)
            
            # Compute loss J(W) and minimize by updating weights - standard logistic loss for model
            log_clf_results = -np.log(classification_res[range(len(curr_batch_of_Y)), curr_batch_of_Y.argmax(axis=1)])
            data_loss = np.sum(log_clf_results) / len(curr_batch_of_Y) #cross entropy loss
            reg_loss = 0.5 * regularization_lambdaa * np.sum(W**2) #Frobenius norm regularization
            loss = data_loss + reg_loss #total_loss
            
            # Compute gradients --- dW = dW_cross_entropy + dW_reg
            grad_desc_arr = classification_res
            grad_desc_arr[range(len(curr_batch_of_Y)), curr_batch_of_Y.argmax(axis=1)] -= 1
            grad_desc_arr /= len(curr_batch_of_Y)
            dW = curr_batch_of_X.T @ grad_desc_arr + regularization_lambdaa * W

            '''Print progress
        if epoch % 10 == 0:
            avg_data_loss = total_data_loss / n
            print(f"Epoch {epoch}, Average Cross-Entropy Loss: {avg_data_loss:.4f}")
            '''
            # Update weights to reiterate to minimize
            W -= best_learn_rate * dW
    
    return W

def predict(W, X):
    Z_Product_XW = X @ W
    return np.argmax(Z_Product_XW, axis=1)

def d_dim_2_p_dim_feature_transform(X, G, b):
    return np.sin(X @ G.T + b)

def get_randomG_uniformb(d, p):
    G = np.random.normal(0, np.sqrt(0.005), (p, d))
    b = np.random.uniform(0, 2*np.pi, p)
    return G, b

# Load the data as given in the question code
train_set = torchvision.datasets.FashionMNIST("./data", download=True)
test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False)

X_train = train_set.data.numpy()
labels_train = train_set.targets.numpy()
X_test = test_set.data.numpy()
labels_test = test_set.targets.numpy()

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]*X_test.shape[2]))
X_train = X_train / 255.0
X_test = X_test / 255.0

# Prepare data by doing one hot encoding
Y_train = encoder_fxn_one_hot(labels_train)
Y_test = encoder_fxn_one_hot(labels_test)

# Train the model using the train() function
regularization_lambdaa = 0.01
W_hat = train(X_train, Y_train, regularization_lambdaa)

# call the predict()
results_arr_training = predict(W_hat, X_train)
results_arr_validation = predict(W_hat, X_test)

# compute the train and val errors
train_error = np.mean(results_arr_training != labels_train)
val_error = np.mean(results_arr_validation != labels_test)

print(f"Training error: {train_error:.4f}")
print(f"Testing error: {val_error:.4f}")
#----------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------#
def train_transformed(X, Y, regularization_lambdaa, G, b, best_learn_rate=0.01, epoch_size=10, batch=256):
    X_transformed = d_dim_2_p_dim_feature_transform(X, G, b)
    return train(X_transformed, Y, regularization_lambdaa, best_learn_rate, epoch_size, batch)

def start_cross_validation_mainloop(X, Y, p_values, regularization_lambdaa):
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    d = X_train.shape[1]
    
    error_arr_training = []
    error_arr_validation = []
    
    for p in p_values:
        G, b = get_randomG_uniformb(d, p)
        
        W_hat = train_transformed(X_train, Y_train, regularization_lambdaa, G, b)
        
        X_train_transformed = d_dim_2_p_dim_feature_transform(X_train, G, b)
        X_val_transformed = d_dim_2_p_dim_feature_transform(X_val, G, b)
        
        results_arr_training = predict(W_hat, X_train_transformed)
        results_validation = predict(W_hat, X_val_transformed)
        
        train_error = np.mean(results_arr_training != Y_train.argmax(axis=1))
        val_error = np.mean(results_validation != Y_val.argmax(axis=1))
        
        error_arr_training.append(train_error)
        error_arr_validation.append(val_error)
    
    return error_arr_training, error_arr_validation

# Main execution cross-validation
p_values = [50, 100, 500, 1000, 1500, 2000, 3000, 4000]
regularization_lambdaa = 0.01

error_arr_training, error_arr_validation = start_cross_validation_mainloop(X_train, Y_train, p_values, regularization_lambdaa)

plt.figure(figsize=(12, 8))
plt.plot(p_values, error_arr_training, label='Train Error', marker='o')
plt.plot(p_values, error_arr_validation, label='Val Error', marker='o')
plt.xlabel('p-order transformed features')
plt.xscale('log')
plt.ylabel('Classification Error')
plt.title('Training and Validation Errors vs #p-features')
plt.legend()
plt.grid(True)
plt.show()

print("All Training errors:", error_arr_training)
print("All Validation errors:", error_arr_validation)


# In[ ]:




