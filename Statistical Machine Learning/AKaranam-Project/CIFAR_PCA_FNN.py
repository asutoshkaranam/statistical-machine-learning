#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt

# Fetch and Normalize the CIFAR-10 dataset
def load_and_preprocess_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    training_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testing_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    x_training_full = training_set.data.reshape(-1, 32*32*3)
    y_training_full = np.array(training_set.targets)
    x_testing_full = testing_set.data.reshape(-1, 32*32*3)
    y_testing_full = np.array(testing_set.targets)
    
    # Filter 'deer' (class 4) and 'horse' (class 7)
    deer_and_horse_mask = np.where((y_training_full == 4) | (y_training_full == 7))[0]
    deer_and_horse_mask_test = np.where((y_testing_full == 4) | (y_testing_full == 7))[0]
    
    x_training_full = x_training_full[deer_and_horse_mask]
    y_training_full = y_training_full[deer_and_horse_mask]
    x_testing_full = x_testing_full[deer_and_horse_mask_test]
    y_testing_full = y_testing_full[deer_and_horse_mask_test]
    
    # Map the labels to binary -1 for horse and 1 for deer
    y_training_full = np.where(y_training_full == 4, 1, -1)
    y_testing_full = np.where(y_testing_full == 4, 1, -1)
    
    return x_training_full, y_training_full, x_testing_full, y_testing_full

# Define the NeuralNetwork Class
class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_layers):
        super(NeuralNetwork, self).__init__()
        layers = []
        prev_size = input_size
        for size in hidden_layers:
            layers.append(torch.nn.Linear(prev_size, size))
            layers.append(torch.nn.ReLU())
            prev_size = size
        layers.append(torch.nn.Linear(prev_size, 1))
        layers.append(torch.nn.Sigmoid())
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Apply PCA and Train the Neural Network
def pca_nn(x_training_full, y_training_full, x_testing_full, y_testing_full, n_components, hidden_layers, epochs=100, batch_size=32):
    pca = PCA(n_components=n_components)
    after_pca_x_train = pca.fit_transform(x_training_full)
    after_pca_x_test = pca.transform(x_testing_full)
    
    model = NeuralNetwork(n_components, hidden_layers)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    tensored_x_train = torch.FloatTensor(after_pca_x_train)
    tensored_y_train = torch.FloatTensor(y_training_full.reshape(-1, 1))
    
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(after_pca_x_train), batch_size):
            batch_x = tensored_x_train[i:i+batch_size]
            batch_y = tensored_y_train[i:i+batch_size]
            
            outputs = model(batch_x)
            loss = criterion(outputs, (batch_y + 1) / 2)  # Adjust labels to [0, 1] as the criterion takes only 0 and 1
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        y_predicted = model(torch.FloatTensor(after_pca_x_test)).numpy()
    
    y_predicted_classes = (y_predicted > 0.5).astype(int)
    y_predicted_classes = np.where(y_predicted_classes == 0, -1, 1)
    
    accuracy = accuracy_score(y_testing_full, y_predicted_classes)
    
    return model, accuracy, y_predicted.flatten(), pca

# Hyperparameter Random search function for tuning
def random_hyperparameter_tuning_function(x_training_full, y_training_full, n_iterations=50):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_accuracy = 0
    best_hyperparams = {}
    
    for _ in range(n_iterations):
        n_components = np.random.randint(50, 600)
        hidden_layers = [np.random.randint(32, 256) for _ in range(np.random.randint(1, 5))]
        epochs = np.random.randint(50, 200)
        batch_size = np.random.choice([32, 64, 128])
        
        cross_validation_score_arr = []
        for idx_training, idx_validation in kf.split(x_training_full):
            x_train_fold, x_val_fold = x_training_full[idx_training], x_training_full[idx_validation]
            y_train_fold, y_val_fold = y_training_full[idx_training], y_training_full[idx_validation]
            
            _, accuracy, _, _ = pca_nn(x_train_fold, y_train_fold, x_val_fold, y_val_fold,
                                       n_components, hidden_layers, epochs, batch_size)
            cross_validation_score_arr.append(accuracy)
        
        average_cross_validation_score = np.mean(cross_validation_score_arr)
        if average_cross_validation_score > best_accuracy:
            best_accuracy = average_cross_validation_score
            best_hyperparams = {
                'n_components': n_components,
                'hidden_layers': hidden_layers,
                'epochs': epochs,
                'batch_size': batch_size
            }
    
    return best_hyperparams, best_accuracy

# Track accuracy as a function of number of epochs for plotting
def track_accuracy_by_epochs(x_training_full, y_training_full, x_testing_full, y_testing_full, hidden_layers, n_components, max_epochs=200):
    # Train and Validation Split of training data
    x_train_subset, x_val, y_train_subset, y_val = train_test_split(x_training_full, y_training_full, test_size=0.2, random_state=42)
    
    # Apply PCA on both the train and validation sets
    pca = PCA(n_components=n_components)
    after_pca_x_train_subset = pca.fit_transform(x_train_subset)
    after_pca_x_validation_subset = pca.transform(x_val)
    after_pca_x_test = pca.transform(x_testing_full)
    
    # Prepare tensors for analyses
    tensored_x_train = torch.FloatTensor(after_pca_x_train_subset)
    tensored_y_train = torch.FloatTensor(y_train_subset.reshape(-1, 1))
    
    # Accuracy measurement and tracking
    train_accrcy_arr = []
    val_accrcy_arr = []
    test_accrcy_arr = []
    
    # Epoch recording
    list_of_epochs_tracking = list(range(0, max_epochs + 1, 10)) or [max_epochs]
    list_of_epochs_tracking = [0] + list_of_epochs_tracking if list_of_epochs_tracking[0] != 0 else list_of_epochs_tracking
    
    for tracked_epoch in list_of_epochs_tracking:
        # Create and train model
        model = NeuralNetwork(n_components, hidden_layers)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Train model to the tracked epoch
        for epoch in range(tracked_epoch):
            model.train()
            outputs = model(tensored_x_train)
            loss = criterion(outputs, (tensored_y_train + 1) / 2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Evaluate model
        model.eval()
        with torch.no_grad():
            # Train accuracy
            train_predicted = model(tensored_x_train).numpy()
            train_pred_classes = (train_predicted > 0.5).astype(int)
            train_pred_classes = np.where(train_pred_classes == 0, -1, 1)
            train_acc = accuracy_score(y_train_subset, train_pred_classes)
            
            # Validation accuracy
            validation_predicted = model(torch.FloatTensor(after_pca_x_validation_subset)).numpy()
            val_pred_classes = (validation_predicted > 0.5).astype(int)
            val_pred_classes = np.where(val_pred_classes == 0, -1, 1)
            val_acc = accuracy_score(y_val, val_pred_classes)
            
            # Test accuracy
            test_predicted = model(torch.FloatTensor(after_pca_x_test)).numpy()
            test_pred_classes = (test_predicted > 0.5).astype(int)
            test_pred_classes = np.where(test_pred_classes == 0, -1, 1)
            test_acc = accuracy_score(y_testing_full, test_pred_classes)
        
        train_accrcy_arr.append(train_acc)
        val_accrcy_arr.append(val_acc)
        test_accrcy_arr.append(test_acc)
    
    return {
        'epochs': list_of_epochs_tracking,
        'train_accrcy_arr': train_accrcy_arr,
        'val_accrcy_arr': val_accrcy_arr,
        'test_accrcy_arr': test_accrcy_arr
    }

# Clone the above function to track the accuracy as a function of n_components for plotting
def track_accuracy_by_components(x_training_full, y_training_full, x_testing_full, y_testing_full, hidden_layers, max_epochs=100):
    # Split training data into train and validation sets
    x_train_subset, x_val, y_train_subset, y_val = train_test_split(x_training_full, y_training_full, test_size=0.2, random_state=42)
    
    # Track accuracies for different PCA components
    principal_component_range = list(range(10, x_training_full.shape[1], max(1, x_training_full.shape[1] // 5)))
    
    train_accrcy_arr = []
    val_accrcy_arr = []
    test_accrcy_arr = []
    
    for n_components in principal_component_range:
        # Apply PCA
        pca = PCA(n_components=n_components)
        after_pca_x_train_subset = pca.fit_transform(x_train_subset)
        after_pca_x_validation_subset = pca.transform(x_val)
        after_pca_x_test = pca.transform(x_testing_full)
        
        # Prepare tensors
        tensored_x_train = torch.FloatTensor(after_pca_x_train_subset)
        tensored_y_train = torch.FloatTensor(y_train_subset.reshape(-1, 1))
        
        # Create and train model
        model = NeuralNetwork(n_components, hidden_layers)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Train model
        for _ in range(max_epochs):
            model.train()
            outputs = model(tensored_x_train)
            loss = criterion(outputs, (tensored_y_train + 1) / 2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Evaluate model
        model.eval()
        with torch.no_grad():
            # Train accuracy
            train_predicted = model(tensored_x_train).numpy()
            train_pred_classes = (train_predicted > 0.5).astype(int)
            train_pred_classes = np.where(train_pred_classes == 0, -1, 1)
            train_acc = accuracy_score(y_train_subset, train_pred_classes)
            
            # Validation accuracy
            validation_predicted = model(torch.FloatTensor(after_pca_x_validation_subset)).numpy()
            val_pred_classes = (validation_predicted > 0.5).astype(int)
            val_pred_classes = np.where(val_pred_classes == 0, -1, 1)
            val_acc = accuracy_score(y_val, val_pred_classes)
            
            # Test accuracy
            test_predicted = model(torch.FloatTensor(after_pca_x_test)).numpy()
            test_pred_classes = (test_predicted > 0.5).astype(int)
            test_pred_classes = np.where(test_pred_classes == 0, -1, 1)
            test_acc = accuracy_score(y_testing_full, test_pred_classes)
        
        train_accrcy_arr.append(train_acc)
        val_accrcy_arr.append(val_acc)
        test_accrcy_arr.append(test_acc)
    
    return {
        'components': principal_component_range,
        'train_accrcy_arr': train_accrcy_arr,
        'val_accrcy_arr': val_accrcy_arr,
        'test_accrcy_arr': test_accrcy_arr
    }

# Define the Plotting functions below
def plot_accuracy_by_epochs(accuracy_data):
    plt.figure(figsize=(10, 6))
    plt.plot(accuracy_data['epochs'], accuracy_data['train_accrcy_arr'], label='Train Accuracy', marker='o')
    plt.plot(accuracy_data['epochs'], accuracy_data['val_accrcy_arr'], label='Validation Accuracy', marker='s')
    plt.plot(accuracy_data['epochs'], accuracy_data['test_accrcy_arr'], label='Test Accuracy', marker='^')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Across Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy_by_components(accuracy_data):
    plt.figure(figsize=(10, 6))
    plt.plot(accuracy_data['components'], accuracy_data['train_accrcy_arr'], label='Train Accuracy', marker='o')
    plt.plot(accuracy_data['components'], accuracy_data['val_accrcy_arr'], label='Validation Accuracy', marker='s')
    plt.plot(accuracy_data['components'], accuracy_data['test_accrcy_arr'], label='Test Accuracy', marker='^')
    plt.xlabel('Number of PCA Components')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Across PCA Components')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Fetch and Normalize the dataset
    x_training_full, y_training_full, x_testing_full, y_testing_full = load_and_preprocess_data()

    # Perform random hyperparameter search to get the best set of hyperparameters
    best_hyperparams, best_cross_validation_accrcy = random_hyperparameter_tuning_function(x_training_full, y_training_full)
    # best_hyperparams = {
    # "n_components": 185,
    # "hidden_layers": [252, 202, 120],
    # "epochs": 128,
    # "batch_size": 32,
    # "lr": 0.001,  # Default Adam learning rate
    # "dropout_rate": 0.3  # Standard dropout rate in the first script
    # }
    print("Best hyperparameters:", best_hyperparams)
    print("Best cross-validation accuracy:", best_cross_validation_accrcy)

    # Train final model with best hyperparameters
    best_model, test_accuracy, y_predicted, pca = pca_nn(x_training_full, y_training_full, x_testing_full, y_testing_full,
                                                    best_hyperparams['n_components'],
                                                    best_hyperparams['hidden_layers'],
                                                    best_hyperparams['epochs'],
                                                    best_hyperparams['batch_size'])

    print("Test accuracy:", test_accuracy)

    # Per-class accuracy
    y_predicted_classes = (y_predicted > 0.5).astype(int)
    y_predicted_classes = np.where(y_predicted_classes == 0, -1, 1)
    deer_accuracy = accuracy_score(y_testing_full[y_testing_full == 1], y_predicted_classes[y_testing_full == 1])
    horse_accuracy = accuracy_score(y_testing_full[y_testing_full == -1], y_predicted_classes[y_testing_full == -1])
    print("Deer accuracy:", deer_accuracy)
    print("Horse accuracy:", horse_accuracy)

    # Plot AUROC
    fpr, tpr, _ = roc_curve(y_testing_full, y_predicted)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Plot cumulative explained variance ratio
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Cumulative Explained Variance Ratio')
    plt.show()

    # Track and plot accuracy by epochs
    epochs_accuracy = track_accuracy_by_epochs(
        x_training_full, y_training_full, x_testing_full, y_testing_full, 
        best_hyperparams['hidden_layers'], 
        best_hyperparams['n_components']
    )
    plot_accuracy_by_epochs(epochs_accuracy)

    # Track and plot accuracy by PCA components
    components_accuracy = track_accuracy_by_components(
        x_training_full, y_training_full, x_testing_full, y_testing_full, 
        best_hyperparams['hidden_layers']
    )
    plot_accuracy_by_components(components_accuracy)

if __name__ == "__main__":
    main()


# In[ ]:




