#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler

# Fetch the MNIST dataset
mnist_dataset_full = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist_dataset_full.data.astype('float32'), mnist_dataset_full.target.astype(int)

# Split the Dataset to train set and test set
X_train_full, X_test = X[:60000], X[60000:]
y_train_full, y_test = y[:60000], y[60000:]

# Extract the samples with digits 3 and 8
mask_train = (y_train_full == 3) | (y_train_full == 8)
mask_test = (y_test == 3) | (y_test == 8)
X_train_full, y_train_full = X_train_full[mask_train], y_train_full[mask_train]
X_test, y_test = X_test[mask_test], y_test[mask_test]

# Map the digit 3 to 1, and digit 8 to -1
y_train_full = np.where(y_train_full == 3, 1, -1)
y_test = np.where(y_test == 3, 1, -1)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)

# Grid Dictionary for Hyperparameter tuning in the GridSearchCV pipeline
hyperparameter_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'kernel': ['rbf', 'poly', 'linear'],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'degree' : [2, 3, 4]
}

# Initialize the SVM classification model
svm = SVC(probability=True, random_state=42)

# Perform ModelSelection using GridSearchCV with K-fold cross-validation
grid_search_piepline = GridSearchCV(svm, hyperparameter_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_piepline.fit(X_train_scaled, y_train_full)

# Identify the best model and its hyperparameters
best_classifier = grid_search_piepline.best_estimator_
best_hyperparams = grid_search_piepline.best_params_

# Evaluate the model on the test set
y_predicted = best_classifier.predict(X_test_scaled)
y_probable = best_classifier.predict_proba(X_test_scaled)[:, 1]

# Compute and print the accuracy and per-class accuracies using confusion matrix
accuracy = accuracy_score(y_test, y_predicted)
confuse_matrix = confusion_matrix(y_test, y_predicted)
perclass_accrcy = confuse_matrix.diagonal() / confuse_matrix.sum(axis=1)

# Display the results
print(f"Best parameters: {best_hyperparams}")
print(f"Overall accuracy: {accuracy:.4f}")
print(f"Per-class accuracy: {perclass_accrcy}")

# Plot ROC curve and indicate the AUROC
fpr, tpr, _ = roc_curve(y_test, y_probable)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Plot accuracy as a function of regularization strength - (C) curve
different_values_of_C = [0.01, 0.1, 1, 10, 100]
train_scores = []
test_scores = []

for C in different_values_of_C:
    svm = SVC(kernel='rbf', C=C, gamma='auto', random_state=42)
    svm.fit(X_train_scaled, y_train_full)
    train_scores.append(svm.score(X_train_scaled, y_train_full))
    test_scores.append(svm.score(X_test_scaled, y_test))

plt.figure()
plt.semilogx(different_values_of_C, train_scores, label='Training accuracy')
plt.semilogx(different_values_of_C, test_scores, label='Test accuracy')
plt.xlabel('Regularization strength (C)')
plt.ylabel('Accuracy')
plt.title('SVM Performance vs Regularization Strength')
plt.legend()
plt.show()

# Plot learning curve for the best SVM
def plot_learning_curve(estimator, X, y, title):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)
    
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Accuracy")
    plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Testing Accuracy")
    
    plt.legend(loc="best")
    plt.show()

# Plot learning curve
plot_learning_curve(best_classifier, X_train_scaled, y_train_full, "Learning Curve (SVM)")


# In[ ]:




