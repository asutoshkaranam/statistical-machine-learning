#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import (
    roc_curve, auc, 
    confusion_matrix, 
    classification_report
)
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Check if GPU is available and set random seed
def set_reproducibility_seeding(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

# Fetch and process the data
def prepare_binary_dataset(transform, binary_classes_needed={4: 1, 7: 0}, train=True):
    # Fetch cifar_dataset using torchvision
    cifar_dataset = datasets.CIFAR10('data', train=train, download=True, transform=transform)
    
    # Filter and map classes to 0/1 or -1/1 -- be sure to change the criterion of optimizer
    mask_indexes = [i for i, (_, label) in enumerate(cifar_dataset) if label in binary_classes_needed]
    binary_version = Subset(cifar_dataset, mask_indexes)
    
    # Remap the output_labels of the dataset to a matrix
    dataset_arr = list(binary_version)
    labelled_dataset_grid = [(img, binary_classes_needed[label]) for img, label in dataset_arr]
    
    return labelled_dataset_grid

# Define a CNN Model with 3-Conv Layers and a classification layer(s)
class BinaryCNN(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(BinaryCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Random Hyperparameter Search Algorithm
def random_hyperparameter_tuning_function(hyperparameter_grid, n_iterations=50):
    keys = list(hyperparameter_grid.keys())
    return [
        {key: random.choice(hyperparameter_grid[key]) for key in keys} 
        for _ in range(n_iterations)
    ]

# Implement a Custom K-Fold Cross-Validation method
def custom_kfold_cross_validation(cifar_dataset, CNN_class, hyperparams, device, n_splits=5):
    k_fold_cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_accrcy_arr = []
    fold_auc_arr = []
    
    X = [item[0] for item in cifar_dataset]
    y = [item[1] for item in cifar_dataset]
    
    for fold, (idx_training, idx_validation) in enumerate(k_fold_cv.split(X), 1):
        training_subset_full = [cifar_dataset[i] for i in idx_training]
        validation_subset_full = [cifar_dataset[i] for i in idx_validation]
        
        training_data_loader = DataLoader(training_subset_full, batch_size=hyperparams['batch_size'], shuffle=True)
        validation_data_loader = DataLoader(validation_subset_full, batch_size=hyperparams['batch_size'], shuffle=False)
        
        model = CNN_class(dropout_rate=hyperparams['dropout_rate']).to(device)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=hyperparams['lr'])
        
        for epoch in range(hyperparams['epochs']):
            model.train()
            for model_feed_input, output_labels in training_data_loader:
                model_feed_input, output_labels = model_feed_input.to(device), output_labels.to(device)
                output_labels = output_labels.float().unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = model(model_feed_input)
                loss = criterion(outputs, output_labels)
                loss.backward()
                optimizer.step()
        
        model.eval()
        validation_correct_predicts, total_validations = 0, 0
        validation_probability_arr, val_labels = [], []
        
        with torch.no_grad():
            for model_feed_input, output_labels in validation_data_loader:
                model_feed_input, output_labels = model_feed_input.to(device), output_labels.to(device)
                output_labels = output_labels.float().unsqueeze(1)
                
                outputs = model(model_feed_input)
                predictions = (outputs > 0.5).float()
                
                validation_correct_predicts += (predictions == output_labels).sum().item()
                total_validations += output_labels.size(0)
                
                validation_probability_arr.extend(outputs.cpu().numpy())
                val_labels.extend(output_labels.cpu().numpy())
        
        fold_accuracy = validation_correct_predicts / total_validations
        fold_accrcy_arr.append(fold_accuracy)
        
        fold_auc = auc(roc_curve(val_labels, validation_probability_arr)[0], 
                       roc_curve(val_labels, validation_probability_arr)[1])
        fold_auc_arr.append(fold_auc)
        
        print(f"Fold {fold}: Accuracy = {fold_accuracy:.4f}, AUC = {fold_auc:.4f}")
    
    return {
        'mean_accuracy': np.mean(fold_accrcy_arr),
        'std_accuracy': np.std(fold_accrcy_arr),
        'mean_auc': np.mean(fold_auc_arr),
        'std_auc': np.std(fold_auc_arr)
    }

#Plotting purpose - find the accuracies as a function of epochs
def train_and_validate_with_tracking(model, training_data_loader, validation_data_loader, testing_data_loader, criterion, optimizer, epochs, device):
    train_accrcy_arr = []
    validation_accrcy_arr = []
    test_accrcy_arr = []
    
    for epoch in range(epochs):
        model.train()
        training_correct_predict, total_training_predicts = 0, 0
        for model_feed_input, output_labels in training_data_loader:
            model_feed_input, output_labels = model_feed_input.to(device), output_labels.to(device)
            output_labels = output_labels.float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(model_feed_input)
            loss = criterion(outputs, output_labels)
            loss.backward()
            optimizer.step()
            
            predictions = (outputs > 0.5).float()
            training_correct_predict += (predictions == output_labels).sum().item()
            total_training_predicts += output_labels.size(0)
        
        train_accuracy = training_correct_predict / total_training_predicts
        train_accrcy_arr.append(train_accuracy)
        
        model.eval()
        validation_correct_predicts, total_validations = 0, 0
        with torch.no_grad():
            for model_feed_input, output_labels in validation_data_loader:
                model_feed_input, output_labels = model_feed_input.to(device), output_labels.to(device)
                output_labels = output_labels.float().unsqueeze(1)
                
                outputs = model(model_feed_input)
                predictions = (outputs > 0.5).float()
                
                validation_correct_predicts += (predictions == output_labels).sum().item()
                total_validations += output_labels.size(0)
        
        val_accuracy = validation_correct_predicts / total_validations
        validation_accrcy_arr.append(val_accuracy)
        
        testing_correct_predictions, total_testing_predictions = 0, 0
        with torch.no_grad():
            for model_feed_input, output_labels in testing_data_loader:
                model_feed_input, output_labels = model_feed_input.to(device), output_labels.to(device)
                output_labels = output_labels.float().unsqueeze(1)
                
                outputs = model(model_feed_input)
                predictions = (outputs > 0.5).float()
                
                testing_correct_predictions += (predictions == output_labels).sum().item()
                total_testing_predictions += output_labels.size(0)
        
        # Calculate test accuracy for this epoch
        test_accuracy = testing_correct_predictions / total_testing_predictions
        test_accrcy_arr.append(test_accuracy)
        
        print(f"Epoch {epoch+1}: Train Accuracy = {train_accuracy:.4f}, Val Accuracy = {val_accuracy:.4f}, Test Accuracy = {test_accuracy:.4f}")
    
    return {
        'train_accrcy_arr': train_accrcy_arr,
        'validation_accrcy_arr': validation_accrcy_arr,
        'test_accrcy_arr': test_accrcy_arr
    }

def learning_curve_plotter(train_accrcy_arr, validation_accrcy_arr, test_accrcy_arr):
    plt.figure(figsize=(12, 7))
    plt.plot(range(1, len(train_accrcy_arr) + 1), train_accrcy_arr, label='Training Accuracy', marker='o')
    plt.plot(range(1, len(validation_accrcy_arr) + 1), validation_accrcy_arr, label='Validation Accuracy', marker='x')
    plt.plot(range(1, len(test_accrcy_arr) + 1), test_accrcy_arr, label='Test Accuracy', marker='^')
    plt.title('Learning Curves: Training, Validation, and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def evaluate_model_with_testset(model, testing_data_loader, device):
    model.eval()
    labels_arr_all = []
    predictions_arr_all = []
    probability_arr_all = []
    
    with torch.no_grad():
        testing_correct_predictions, total_testing_predictions = 0, 0
        for model_feed_input, output_labels in testing_data_loader:
            model_feed_input, output_labels = model_feed_input.to(device), output_labels.to(device)
            output_labels = output_labels.float().unsqueeze(1)
            
            outputs = model(model_feed_input)
            predictions = (outputs > 0.5).float()
            
            testing_correct_predictions += (predictions == output_labels).sum().item()
            total_testing_predictions += output_labels.size(0)
            
            labels_arr_all.extend(output_labels.cpu().numpy())
            predictions_arr_all.extend(predictions.cpu().numpy())
            probability_arr_all.extend(outputs.cpu().numpy())
    
    test_accuracy = testing_correct_predictions / total_testing_predictions
    
    labels_arr_all = np.array(labels_arr_all)
    predictions_arr_all = np.array(predictions_arr_all)
    probability_arr_all = np.array(probability_arr_all)
    
    fpr, tpr, _ = roc_curve(labels_arr_all, probability_arr_all)
    roc_auc = auc(fpr, tpr)
    
    confuse_matrix = confusion_matrix(labels_arr_all, predictions_arr_all)
    class_report = classification_report(labels_arr_all, predictions_arr_all)
    
    return {
        'accuracy': test_accuracy,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'confusion_matrix': confuse_matrix,
        'classification_report': class_report
    }

def display_confusion_matrix(confuse_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(confuse_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

def compute_accuracy_classwise(model, data_loader, device):
    model.eval()
    correct_classes = [0, 0]
    total_classes = [0, 0]
    
    with torch.no_grad():
        for model_feed_input, output_labels in data_loader:
            model_feed_input, output_labels = model_feed_input.to(device), output_labels.to(device)
            output_labels = output_labels.float().unsqueeze(1)
            
            outputs = model(model_feed_input)
            predictions = (outputs > 0.5).float()
            
            for pred, label in zip(predictions, output_labels):
                class_idx = int(label.item())
                total_classes[class_idx] += 1
                if pred.item() == label.item():
                    correct_classes[class_idx] += 1
    
    per_class_accuracy = [
        correct / total if total > 0 else 0 
        for correct, total in zip(correct_classes, total_classes)
    ]
    
    return per_class_accuracy, total_classes

def compute_learning_rate_vs_accuracy(
    base_lr=0.0001, 
    max_lr=0.1, 
    num_lrs=5, 
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset_full = prepare_binary_dataset(transform, train=True)
    test_dataset_full = prepare_binary_dataset(transform, train=False)

    train_size = int(0.8 * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size
    train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

    hyperparams = {
        'batch_size': 16, 
        'dropout_rate': 0.1, 
        'epochs': 25
    }

    learning_rates = np.logspace(
        np.log10(base_lr), 
        np.log10(max_lr), 
        num=num_lrs
    )

    training_data_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
    validation_data_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'], shuffle=False)
    testing_data_loader = DataLoader(test_dataset_full, batch_size=hyperparams['batch_size'], shuffle=False)

    # Track accuracies
    train_accrcy_arr = []
    validation_accrcy_arr = []
    test_accrcy_arr = []

    # Iterate through different learning rates
    for lr in learning_rates:
        # Reset model for each learning rate
        model = BinaryCNN(dropout_rate=hyperparams['dropout_rate']).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training loop
        for epoch in range(hyperparams['epochs']):
            model.train()
            for model_feed_input, output_labels in training_data_loader:
                model_feed_input, output_labels = model_feed_input.to(device), output_labels.to(device)
                output_labels = output_labels.float().unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = model(model_feed_input)
                loss = criterion(outputs, output_labels)
                loss.backward()
                optimizer.step()

        # Evaluate model
        model.eval()
        
        # Calculate accuracies
        def calculate_accuracy(data_loader):
            correct, total = 0, 0
            with torch.no_grad():
                for model_feed_input, output_labels in data_loader:
                    model_feed_input, output_labels = model_feed_input.to(device), output_labels.to(device)
                    output_labels = output_labels.float().unsqueeze(1)
                    
                    outputs = model(model_feed_input)
                    predictions = (outputs > 0.5).float()
                    
                    correct += (predictions == output_labels).sum().item()
                    total += output_labels.size(0)
            return correct / total

        # Store accuracies
        train_accrcy_arr.append(calculate_accuracy(training_data_loader))
        validation_accrcy_arr.append(calculate_accuracy(validation_data_loader))
        test_accrcy_arr.append(calculate_accuracy(testing_data_loader))

    # Plot learning rates vs accuracies
    plt.figure(figsize=(10, 6))
    plt.semilogx(learning_rates, train_accrcy_arr, marker='o', label='Training Accuracy')
    plt.semilogx(learning_rates, validation_accrcy_arr, marker='x', label='Validation Accuracy')
    plt.semilogx(learning_rates, test_accrcy_arr, marker='^', label='Test Accuracy')
    plt.title('Learning Rate vs Accuracy')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return learning_rates, train_accrcy_arr, validation_accrcy_arr, test_accrcy_arr
	
def main():
    set_reproducibility_seeding()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Prepare full cifar_dataset
    train_dataset_full = prepare_binary_dataset(transform, train=True)
    test_dataset_full = prepare_binary_dataset(transform, train=False)

    # Hyperparameter ranges
    hyperparameter_grid = {
        "lr": [0.0001, 0.001, 0.01, 0.1],
        "batch_size": [16, 32, 64],
        "dropout_rate": [0.1, 0.2, 0.3, 0.4],
        "epochs": [15, 20, 25, 30]
    }

    # Hyperparameter search with custom K-Fold Cross-Validation
    best_average_accrcy = 0
    best_hyperparams = None

    for hyperparams in random_hyperparameter_tuning_function(hyperparameter_grid, n_iterations=10):
        # Perform K-Fold Cross-Validation
        cross_validation_results = custom_kfold_cross_validation(
            train_dataset_full, 
            BinaryCNN, 
            hyperparams, 
            device
        )
        
        # Update best hyperparameters
        if cross_validation_results['mean_accuracy'] > best_average_accrcy:
            best_average_accrcy = cross_validation_results['mean_accuracy']
            best_hyperparams = hyperparams
    
    print("\nBest Hyperparameters:", best_hyperparams)
    print(f"Best Mean Cross-Validation Accuracy: {best_average_accrcy:.4f}")

    # Prepare final model using full training data and best hyperparameters
    train_size = int(0.8 * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size
    train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

    # Create data loaders
    training_data_loader = DataLoader(train_dataset, batch_size=best_hyperparams['batch_size'], shuffle=True)
    validation_data_loader = DataLoader(val_dataset, batch_size=best_hyperparams['batch_size'], shuffle=False)
    testing_data_loader = DataLoader(test_dataset_full, batch_size=best_hyperparams['batch_size'], shuffle=False)

    # Final model training with tracking
    final_model = BinaryCNN(dropout_rate=best_hyperparams['dropout_rate']).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(final_model.parameters(), lr=best_hyperparams['lr'])

    # Train and track metrics
    training_metrics = train_and_validate_with_tracking(
        final_model, 
        training_data_loader, 
        validation_data_loader, 
        testing_data_loader,  # Added testing_data_loader here
        criterion, 
        optimizer, 
        best_hyperparams['epochs'], 
        device
    )

    # Plot learning curves
    learning_curve_plotter(
        training_metrics['train_accrcy_arr'], 
        training_metrics['validation_accrcy_arr'],
        training_metrics['test_accrcy_arr']  # Added test accuracies
    )

    # Test the final model
    test_metrics = evaluate_model_with_testset(final_model, testing_data_loader, device)
	
	# Calculate per-class accuracy
    per_class_acc, class_totals = compute_accuracy_classwise(final_model, testing_data_loader, device)
    
    # Print per-class accuracy
    print("\nPer-Class Accuracy:")
    class_labels = ['Negative (Class 0)', 'Positive (Class 1)']
    for label, acc, total in zip(class_labels, per_class_acc, class_totals):
        print(f"{label}: {acc:.4f} (Samples: {total})")

    # Print test results
    print("\nTest Results:")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"ROC AUC: {test_metrics['roc_auc']:.4f}")
    print("\nClassification Report:")
    print(test_metrics['classification_report'])

    # Visualizations
    # ROC Curve
    plt.figure(figsize=(10, 6))
    plt.plot(test_metrics['fpr'], test_metrics['tpr'], color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {test_metrics["roc_auc"]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Confusion Matrix
    display_confusion_matrix(test_metrics['confusion_matrix'])
	
	# Learning Rate vs Accuracy Analysis
    compute_learning_rate_vs_accuracy()

if __name__ == "__main__":
    main()


# In[ ]:




