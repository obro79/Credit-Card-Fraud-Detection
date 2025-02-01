import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import zipfile
import os
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
from visualizations import (
    plot_correlation_heatmap,
    plot_amount_distributions,
    plot_time_analysis,
    plot_class_balance,
    plot_feature_importance,
    plot_roc_curve,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_transaction_velocity,
    plot_interaction_features,
    generate_model_visualizations
)

def save_model(model, model_name):
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model to save
        model_name (str): Name of the model for the file
    """
    save_path = f'saved_models/{model_name}'
    joblib.dump(model, f'{save_path}.pkl')
    print(f"Model saved: {model_name}")

def load_model(model_name):
    """
    Load a trained model from disk.
    
    Args:
        model_name (str): Name of the model to load
        
    Returns:
        object: Loaded model or None if not found
    """
    model_path = f'saved_models/{model_name}.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def load_data(zip_path='data/creditcard.csv.zip'):
    """
    Load and prepare the credit card fraud dataset from a zip file.
    
    Args:
        zip_path (str): Path to the zipped CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    # Check if zip file exists
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Dataset not found at {zip_path}")
    
    # Extract and read the CSV file from the zip
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get the name of the CSV file inside the zip
        csv_name = zip_ref.namelist()[0]
        # Read the CSV directly from the zip file
        with zip_ref.open(csv_name) as csv_file:
            df = pd.read_csv(csv_file)
    
    print(f"Dataset loaded successfully with shape: {df.shape}")
    return df

def analyze_data(df):
    """
    Perform comprehensive exploratory data analysis with visualizations.
    
    Args:
        df (pd.DataFrame): Dataset to analyze
    """
    # Create visualizations directory
    os.makedirs('visualizations', exist_ok=True)
    
    print("\n=== Dataset Overview ===")
    print(f"Total transactions: {len(df)}")
    print(f"Number of features: {df.shape[1] - 1}")
    
    # Check missing values
    missing_values = df.isnull().sum()
    if missing_values.any():
        print("\nMissing Values:")
        print(missing_values[missing_values > 0])
    else:
        print("\nNo missing values found in the dataset.")
    
    # Class distribution
    fraud_count = df['Class'].sum()
    normal_count = len(df) - fraud_count
    fraud_percentage = (fraud_count / len(df)) * 100
    
    print("\n=== Class Distribution ===")
    print(f"Normal transactions: {normal_count} ({100 - fraud_percentage:.2f}%)")
    print(f"Fraudulent transactions: {fraud_count} ({fraud_percentage:.2f}%)")
    
    # Generate all EDA visualizations
    plot_correlation_heatmap(df)
    plot_amount_distributions(df)
    plot_time_analysis(df)
    
    # Transaction amount analysis
    print("\n=== Transaction Amount Statistics ===")
    print(df.groupby('Class')['Amount'].describe())

def engineer_features(df):
    """
    Create new features to improve fraud detection.
    
    Args:
        df (pd.DataFrame): Original dataset
        
    Returns:
        pd.DataFrame: Dataset with new features
    """
    # Copy the dataframe to avoid modifying the original
    df_new = df.copy()
    
    # Convert Time to hour of day
    df_new['Hour'] = df_new['Time'].apply(lambda x: (x / 3600) % 24)
    
    # Create amount-based features
    df_new['Amount_Log'] = np.log1p(df_new['Amount'])
    
    # Calculate transaction velocity (number of transactions in the last hour)
    df_new['Trans_Velocity'] = df_new.groupby(df_new['Time'] // 3600)['Time'].transform('count')
    
    # V1-V28 are already processed features from PCA
    # Create interaction features between some of the most important V features
    df_new['V1*V2'] = df_new['V1'] * df_new['V2']
    df_new['V3*V4'] = df_new['V3'] * df_new['V4']
    
    return df_new

def preprocess_data(df):
    """
    Preprocess the credit card dataset including feature engineering,
    scaling, and handling class imbalance.
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        tuple: X_train, X_valid, X_test, y_train, y_valid, y_test, scaler, df_processed
    """
    # Engineer features
    df_processed = engineer_features(df)
    
    # Separate features and target
    X = df_processed.drop(['Class', 'Time'], axis=1)  # Remove Time as we've engineered Hour feature
    y = df_processed['Class']
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    # Second split: separate validation set from remaining data
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance using SMOTE (only on training data)
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    # Visualize class balance before and after SMOTE
    plot_class_balance(y_train, y_train_balanced)
    
    print("\n=== Data Split Information ===")
    print(f"Training set shape: {X_train_balanced.shape}")
    print(f"Validation set shape: {X_valid_scaled.shape}")
    print(f"Testing set shape: {X_test_scaled.shape}")
    
    # Save the scaler for future use
    joblib.dump(scaler, 'saved_models/scaler.pkl')
    print("Saved StandardScaler to saved_models/scaler.pkl")
    
    return (X_train_balanced, X_valid_scaled, X_test_scaled,
            y_train_balanced, y_valid, y_test, scaler, df_processed)

def evaluate_model(model, X, y, model_name="Model", feature_names=None):
    """
    Evaluate a model's performance using multiple metrics.
    
    Args:
        model: Trained model
        X: Features
        y: True labels
        model_name (str): Name of the model for printing
        feature_names: List of feature names for importance plot
    
    Returns:
        float: ROC-AUC score
    """
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    print(f"\n=== {model_name} Performance ===")
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    
    # Generate performance visualizations
    plot_confusion_matrix(y, y_pred)
    plot_roc_curve(y, y_pred_proba)
    plot_precision_recall_curve(y, y_pred_proba)
    
    if feature_names is not None:
        plot_feature_importance(model, feature_names)
    
    auc = roc_auc_score(y, y_pred)
    print(f"\nROC-AUC Score: {auc:.4f}")
    
    return auc

def optimize_xgboost(X_train, X_valid, y_train, y_valid):
    """
    Optimize XGBoost model with best parameters from previous experiments.
    
    Args:
        X_train, X_valid, y_train, y_valid: Training and validation data
    
    Returns:
        xgb.XGBClassifier: Optimized XGBoost model
    """
    model_name = 'optimized_xgboost'
    model = load_model(model_name)
    
    if model is None:
        print("\nTraining optimized XGBoost...")
        
        # Best parameters from previous experiments
        best_params = {
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'min_child_weight': 1,
            'objective': 'binary:logistic',
            'random_state': 42
        }
        
        model = xgb.XGBClassifier(**best_params)
        model.fit(X_train, y_train)
        save_model(model, model_name)
    else:
        print("\nLoaded existing optimized XGBoost model")
    
    # Get feature names for importance plot
    feature_names = [f'V{i}' for i in range(1, 29)]
    feature_names.extend(['Hour', 'Amount', 'Amount_Log', 'Trans_Velocity', 'V1*V2', 'V3*V4'])
    
    evaluate_model(model, X_valid, y_valid, "Optimized XGBoost", feature_names)
    
    # Save as xgboost_best.pkl for app.py
    save_model(model, 'xgboost_best')
    return model

def main():
    """
    Main function to orchestrate the credit card fraud detection pipeline.
    """
    print("Starting Credit Card Fraud Detection System...")
    
    try:
        # Create saved_models directory if it doesn't exist
        os.makedirs('saved_models', exist_ok=True)
        
        # Load the dataset
        df = load_data()
        
        # Perform EDA
        analyze_data(df)
        
        # Preprocess the data
        (X_train, X_valid, X_test,
         y_train, y_valid, y_test,
         scaler, df_processed) = preprocess_data(df)
        
        # Train optimized XGBoost model
        best_model = optimize_xgboost(X_train, X_valid, y_train, y_valid)
        
        # Final evaluation on test set
        print("\n=== Final Evaluation on Test Set ===")
        feature_names = [f'V{i}' for i in range(1, 29)]
        feature_names.extend(['Hour', 'Amount', 'Amount_Log', 'Trans_Velocity', 'V1*V2', 'V3*V4'])
        final_auc = evaluate_model(best_model, X_test, y_test, "Best Model (XGBoost)", feature_names)
        
        # Generate transaction velocity and interaction features visualizations
        plot_transaction_velocity(df_processed)
        plot_interaction_features(df_processed)
        
        print("\nModel training and evaluation completed successfully!")
        print(f"Final ROC-AUC Score on Test Set: {final_auc:.4f}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()