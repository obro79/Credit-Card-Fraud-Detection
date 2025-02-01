import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
import joblib
import os
from datetime import datetime

def plot_correlation_heatmap(data, target='Class', output_path='visualizations/correlation_heatmap.png'):
    """
    Plot correlation heatmap for features with target variable.
    
    Args:
        data: DataFrame containing features and target
        target: Name of target variable
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    correlations = data.corr()[target].sort_values()
    
    # Get top and bottom 10 correlations
    top_features = pd.concat([correlations.head(10), correlations.tail(10)])
    correlation_matrix = data[top_features.index].corr()
    
    # Plot heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Correlation heatmap saved to {output_path}")

def plot_amount_distributions(data, output_path='visualizations/amount_distributions.png'):
    """
    Plot KDE plots for Amount and log(Amount).
    
    Args:
        data: DataFrame containing Amount and Class columns
        output_path: Path to save the plot
    """
    plt.figure(figsize=(15, 5))
    
    # Original Amount Distribution
    plt.subplot(1, 2, 1)
    for label in [0, 1]:
        sns.kdeplot(data=data[data['Class'] == label]['Amount'], 
                   label=f"{'Fraud' if label == 1 else 'Normal'}")
    plt.title('Transaction Amount Distribution')
    plt.xlabel('Amount')
    plt.ylabel('Density')
    plt.legend()
    
    # Log Amount Distribution
    plt.subplot(1, 2, 2)
    for label in [0, 1]:
        sns.kdeplot(data=np.log1p(data[data['Class'] == label]['Amount']), 
                   label=f"{'Fraud' if label == 1 else 'Normal'}")
    plt.title('Log Transaction Amount Distribution')
    plt.xlabel('Log(Amount + 1)')
    plt.ylabel('Density')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Amount distributions saved to {output_path}")

def plot_time_analysis(data, output_path='visualizations/time_analysis.png'):
    """
    Plot time-based analysis of transactions.
    
    Args:
        data: DataFrame containing Time and Class columns
        output_path: Path to save the plot
    """
    plt.figure(figsize=(15, 5))
    
    # Transaction count over time
    plt.subplot(1, 2, 1)
    data['Hour'] = data['Time'] / 3600  # Convert Time to hours
    plt.hist(data['Hour'], bins=48, alpha=0.5, label='All Transactions')
    plt.hist(data[data['Class'] == 1]['Hour'], bins=48, alpha=0.5, label='Fraudulent')
    plt.title('Transaction Distribution Over Time')
    plt.xlabel('Hours Since Start')
    plt.ylabel('Number of Transactions')
    plt.legend()
    
    # Fraud rate over time
    plt.subplot(1, 2, 2)
    fraud_rate = data.groupby(data['Hour'].astype(int))['Class'].mean()
    plt.plot(fraud_rate.index, fraud_rate.values)
    plt.title('Fraud Rate Over Time')
    plt.xlabel('Hours Since Start')
    plt.ylabel('Fraud Rate')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Time analysis plots saved to {output_path}")

def plot_class_balance(original_y, resampled_y, output_path='visualizations/class_balance.png'):
    """
    Plot class distribution before and after SMOTE.
    
    Args:
        original_y: Original target variable
        resampled_y: Resampled target variable after SMOTE
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Original distribution
    plt.subplot(1, 2, 1)
    sns.countplot(x=original_y)
    plt.title('Original Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    
    # SMOTE distribution
    plt.subplot(1, 2, 2)
    sns.countplot(x=resampled_y)
    plt.title('Class Distribution After SMOTE')
    plt.xlabel('Class')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Class balance visualization saved to {output_path}")

def plot_feature_importance(model, feature_names, output_path='visualizations/feature_importance.png'):
    """
    Plot feature importance for XGBoost model.
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        output_path: Path to save the plot
    """
    # Get feature importance
    importance = model.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importance)[::-1]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot feature importance
    plt.bar(range(len(importance)), importance[indices])
    
    # Add feature names
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha='right')
    
    # Customize plot
    plt.title('Feature Importance (XGBoost)')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path)
    plt.close()
    print(f"Feature importance plot saved to {output_path}")

def plot_roc_curve(y_true, y_pred_proba, output_path='visualizations/roc_curve.png'):
    """
    Plot ROC curve with AUC score.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        output_path: Path to save the plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()
    print(f"ROC curve saved to {output_path}")

def plot_confusion_matrix(y_true, y_pred, output_path='visualizations/confusion_matrix.png'):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Path to save the plot
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    
    # Customize plot
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix plot saved to {output_path}")

def plot_precision_recall_curve(y_true, y_pred_proba, output_path='visualizations/precision_recall_curve.png'):
    """
    Plot precision-recall curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        output_path: Path to save the plot
    """
    # Compute precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Plot precision-recall curve
    plt.plot(recall, precision, label='Precision-Recall curve')
    
    # Customize plot
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path)
    plt.close()
    print(f"Precision-recall curve saved to {output_path}")

def plot_transaction_velocity(data, output_path='visualizations/transaction_velocity.png'):
    """
    Plot transaction velocity analysis.
    
    Args:
        data: DataFrame containing Time, Trans_Velocity, and Class columns
        output_path: Path to save the plot
    """
    plt.figure(figsize=(15, 5))
    
    # Transaction velocity over time
    plt.subplot(1, 2, 1)
    plt.scatter(data[data['Class'] == 0]['Time'] / 3600, 
               data[data['Class'] == 0]['Trans_Velocity'],
               alpha=0.5, label='Normal', s=1)
    plt.scatter(data[data['Class'] == 1]['Time'] / 3600,
               data[data['Class'] == 1]['Trans_Velocity'],
               alpha=0.5, label='Fraud', s=5)
    plt.title('Transaction Velocity Over Time')
    plt.xlabel('Hours Since Start')
    plt.ylabel('Transaction Velocity')
    plt.legend()
    
    # Velocity distribution
    plt.subplot(1, 2, 2)
    sns.boxplot(x='Class', y='Trans_Velocity', data=data)
    plt.title('Transaction Velocity Distribution by Class')
    plt.xlabel('Class (0=Normal, 1=Fraud)')
    plt.ylabel('Transaction Velocity')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Transaction velocity plots saved to {output_path}")

def plot_interaction_features(data, output_path='visualizations/interaction_features.png'):
    """
    Plot distributions of interaction features.
    
    Args:
        data: DataFrame containing interaction features and Class
        output_path: Path to save the plot
    """
    plt.figure(figsize=(15, 5))
    
    # V1*V2 interaction
    plt.subplot(1, 2, 1)
    for label in [0, 1]:
        sns.kdeplot(data=data[data['Class'] == label]['V1*V2'],
                   label=f"{'Fraud' if label == 1 else 'Normal'}")
    plt.title('V1*V2 Interaction Distribution')
    plt.xlabel('V1*V2')
    plt.ylabel('Density')
    plt.legend()
    
    # V3*V4 interaction
    plt.subplot(1, 2, 2)
    for label in [0, 1]:
        sns.kdeplot(data=data[data['Class'] == label]['V3*V4'],
                   label=f"{'Fraud' if label == 1 else 'Normal'}")
    plt.title('V3*V4 Interaction Distribution')
    plt.xlabel('V3*V4')
    plt.ylabel('Density')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Interaction features plots saved to {output_path}")

def generate_model_visualizations(model_path='saved_models/xgboost_best.pkl', 
                                data=None, X_test=None, y_test=None, 
                                feature_names=None, resampled_y=None):
    """
    Generate all model performance visualizations.
    
    Args:
        model_path: Path to the saved model
        data: Original DataFrame with all features
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
        resampled_y: Resampled target variable after SMOTE
    """
    # Create visualizations directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Load model
    model = joblib.load(model_path)
    
    # Generate visualizations
    if data is not None:
        plot_correlation_heatmap(data)
        plot_amount_distributions(data)
        plot_time_analysis(data)
        
        if 'Trans_Velocity' in data.columns:
            plot_transaction_velocity(data)
        if 'V1*V2' in data.columns and 'V3*V4' in data.columns:
            plot_interaction_features(data)
    
    if feature_names is not None:
        plot_feature_importance(model, feature_names)
    
    if y_test is not None and resampled_y is not None:
        plot_class_balance(y_test, resampled_y)
    
    if X_test is not None and y_test is not None:
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Generate plots
        plot_confusion_matrix(y_test, y_pred)
        plot_precision_recall_curve(y_test, y_pred_proba)
        plot_roc_curve(y_test, y_pred_proba)