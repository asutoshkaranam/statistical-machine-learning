import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, learning_curve
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Fetch the dataset
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.astype('float32')
y = mnist.target.astype(int)

# Separate the dataset to train set and test set
X_train_full, X_test = X[:60000], X[60000:]
y_train_full, y_test = y[:60000], y[60000:]

# Extract the binary classes 3 and 8
mask_train = (y_train_full == 3) | (y_train_full == 8)
mask_test = (y_test == 3) | (y_test == 8)
X_train_full, y_train_full = X_train_full[mask_train], y_train_full[mask_train]
X_test, y_test = X_test[mask_test], y_test[mask_test]

# Assign the digit 3 to 1 and digit 8 to -1
y_train_full = np.where(y_train_full == 3, 1, -1)
y_test = np.where(y_test == 3, 1, -1)

# Normalize the features
scaler = StandardScaler()
X_train_full_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)

# Grid-Dictionary to have all different hyperparameter for tuning
param_grid = {
    'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'penalty': ['l1', 'l2', None],
    'solver': ['liblinear', 'lbfgs', 'saga'],
    'max_iter': [100, 500, 1000],
    'tol': [1e-4, 1e-3, 1e-2]
}

# Initialize the Logistic regression Model and Start tuning with 5-fold cross-validation
classification_model = LogisticRegression(random_state=42)
hyperparam_grid_for_tuning = GridSearchCV(
    estimator=classification_model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
    verbose=1
)

# Fit to learn different models
hyperparam_grid_for_tuning.fit(X_train_full_scaled, y_train_full)

# Get the Best model with the best set of hyperparameters in terms of accuracy
best_set_of_hyperparams = hyperparam_grid_for_tuning.best_params_
best_learnt_model = hyperparam_grid_for_tuning.best_estimator_
print(f"Best Hyperparameters: {best_set_of_hyperparams}")

# Evaluate on the test set
y_predicted = best_learnt_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_predicted)
print(f"Test Accuracy (Best Model): {test_accuracy:.4f}")

# Using Confusion matrix, calculate the per-class accuracies and print
confuse_matrix = confusion_matrix(y_test, y_predicted)
class_accuracies = confuse_matrix.diagonal() / confuse_matrix.sum(axis=1)
print(f"Per-class Accuracy for digit '3' Regularized : {class_accuracies[0]:.4f}")
print(f"Per-class Accuracy for digit '8' Regularized : {class_accuracies[1]:.4f}")

##----------------------------------------------------------------------------------------------------------------------------##

# Initialize an unregularized LogisticRegression model
unreg_classification_model = LogisticRegression(C=1000, penalty=None, solver='lbfgs', max_iter=1000, random_state=42)
unreg_classification_model.fit(X_train_full_scaled, y_train_full)

# Evaluate the unregularized LogisticRegression model on the test set
y_predicted_unreg = unreg_classification_model.predict(X_test_scaled)
test_accuracy_unreg = accuracy_score(y_test, y_predicted_unreg)
print(f"\nTest Accuracy (Unregularized Model): {test_accuracy_unreg:.4f}")

# Using Confusion matrix for unregularized model, calculate the per-class accuracies and print
confuse_matrix_unreg = confusion_matrix(y_test, y_predicted_unreg)
class_accuracies_unreg = confuse_matrix_unreg.diagonal() / confuse_matrix_unreg.sum(axis=1)
print(f"Per-class Accuracy for digit '3' Unregularized : {class_accuracies_unreg[0]:.4f}")
print(f"Per-class Accuracy for digit '8' Unregularized : {class_accuracies_unreg[1]:.4f}")

# Plot the ROC curve and Indicate the AUROC for both Regularized and UnRegularized models
plt.figure(figsize=(10, 8))

# ROC Curve for the identified best regularized model
y_probable_regularized = best_learnt_model.decision_function(X_test_scaled)
fpr_reg, tpr_reg, _ = roc_curve(y_test, y_probable_regularized)
auroc_reg = roc_auc_score(y_test, y_probable_regularized)

# ROC for the unregularized model
y_probable_unregularized = unreg_classification_model.decision_function(X_test_scaled)
fpr_unreg, tpr_unreg, _ = roc_curve(y_test, y_probable_unregularized)
auroc_unreg = roc_auc_score(y_test, y_probable_unregularized)

plt.plot(fpr_reg, tpr_reg, label=f'Regularized Model (AUROC = {auroc_reg:.4f})')
plt.plot(fpr_unreg, tpr_unreg, label=f'Unregularized Model (AUROC = {auroc_unreg:.4f})')
plt.plot([0, 1], [0, 1], 'k--', lw=0.7)
plt.title('Receiver Operating Characteristic Curves')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# Plot the learning curve
def plot_learning_curve(estimator, X, y, title):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy'
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    
    plt.figure(figsize=(12, 8))
    plt.title(title)
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Accuracy")
    plt.grid(True)
    
    # Learning curves for best regularized model
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Accuracy (Regularized)")
    plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Testing Accuracy (Regularized)")
    
    # Learning curves for unregularized model
    train_sizes_unreg, train_scores_unreg, val_scores_unreg = learning_curve(
        unreg_classification_model, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy'
    )
    
    train_scores_mean_unreg = np.mean(train_scores_unreg, axis=1)
    val_scores_mean_unreg = np.mean(val_scores_unreg, axis=1)
    
    plt.plot(train_sizes_unreg, train_scores_mean_unreg, 'o--', color="b", label="Training Accuracy (Unregularized)")
    plt.plot(train_sizes_unreg, val_scores_mean_unreg, 'o--', color="m", label="Testing Accuracy (Unregularized)")
    
    plt.legend(loc="best")
    plt.ylim(0.8, 1.01)
    plt.show()

plot_learning_curve(best_learnt_model, X_train_full_scaled, y_train_full, "Learning Curves for Logistic Regression Models")

# Plot train and test accuracy vs regularization strength
C_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
train_accuracies_arr = []
test_accuracies_arr = []
train_accuracies_unreg_arr = []
test_accuracies_unreg_arr = []

# Get the best parameters, excluding 'C'
rest_of_hyperparameters = {k: v for k, v in best_set_of_hyperparams.items() if k != 'C'}

for C in C_values:
    # Regularized model
    model = LogisticRegression(C=C, random_state=42, **rest_of_hyperparameters)
    model.fit(X_train_full_scaled, y_train_full)
    train_accuracies_arr.append(model.score(X_train_full_scaled, y_train_full))
    test_accuracies_arr.append(model.score(X_test_scaled, y_test))

# Unregularized model with constant accuracies
for _ in C_values:
    train_accuracies_unreg_arr.append(unreg_classification_model.score(X_train_full_scaled, y_train_full))
    test_accuracies_unreg_arr.append(unreg_classification_model.score(X_test_scaled, y_test))

plt.figure(figsize=(12, 8))
plt.semilogx(C_values, train_accuracies_arr, 'b-o', label='Training Accuracy (Best Regularized)')
plt.semilogx(C_values, test_accuracies_arr, 'r-o', label='Test Accuracy (Best Regularized)')
plt.semilogx(C_values, train_accuracies_unreg_arr, 'g--o', label='Training Accuracy (Unregularized)')
plt.semilogx(C_values, test_accuracies_unreg_arr, 'm--o', label='Test Accuracy (Unregularized)')
plt.title('Training and Test Accuracy vs Regularization Strength')
plt.xlabel('Regularization Strength (C)')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.show()