"""
Machine Learning Model Training for Injury Recovery Prediction
Implements Random Forest and XGBoost models
"""
import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from xgboost import XGBRegressor, XGBClassifier

# Configure matplotlib for headless environments (Render deployment)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from utils.data_preprocessing import load_data, preprocess_data


def train_recovery_models(X, y):
    """
    Train Random Forest and XGBoost models for recovery time prediction
    
    Args:
        X: Feature matrix
        y: Target variable (recovery days)
        
    Returns:
        Dictionary with trained models and metrics
    """
    # Split data - ensure float32 for XGBoost compatibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Convert to float32 for XGBoost
    X_train_xgb = X_train.astype(np.float32)
    X_test_xgb = X_test.astype(np.float32)
    y_train_xgb = y_train.astype(np.float32)
    
    print("Training Random Forest for recovery prediction...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    print("Training XGBoost for recovery prediction...")
    xgb_model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        tree_method='hist',  # More stable for deployment
        enable_categorical=False
    )
    xgb_model.fit(X_train_xgb, y_train_xgb)
    xgb_pred = xgb_model.predict(X_test_xgb)
    
    # Calculate metrics
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_r2 = r2_score(y_test, rf_pred)
    
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    xgb_r2 = r2_score(y_test, xgb_pred)
    
    print(f"\nRandom Forest - RMSE: {rf_rmse:.2f}, R²: {rf_r2:.4f}")
    print(f"XGBoost - RMSE: {xgb_rmse:.2f}, R²: {xgb_r2:.4f}")
    
    return {
        'rf_model': rf_model,
        'xgb_model': xgb_model,
        'rf_rmse': rf_rmse,
        'rf_r2': rf_r2,
        'xgb_rmse': xgb_rmse,
        'xgb_r2': xgb_r2,
        'test_data': (X_test, y_test),
        'predictions': {'rf': rf_pred, 'xgb': xgb_pred}
    }


def train_setback_models(X, y):
    """
    Train Random Forest and XGBoost models for setback risk prediction
    
    Args:
        X: Feature matrix
        y: Target variable (setback indicator)
        
    Returns:
        Dictionary with trained models and metrics
    """
    # Split data - ensure float32 for XGBoost compatibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Convert to float32 for XGBoost
    X_train_xgb = X_train.astype(np.float32)
    X_test_xgb = X_test.astype(np.float32)
    
    print("\nTraining Random Forest for setback prediction...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)
    
    print("Training XGBoost for setback prediction...")
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        tree_method='hist',  # More stable for deployment
        enable_categorical=False
    )
    xgb_model.fit(X_train_xgb, y_train)
    xgb_pred = xgb_model.predict(X_test_xgb)
    xgb_proba = xgb_model.predict_proba(X_test_xgb)
    
    # Calculate metrics
    rf_accuracy = accuracy_score(y_test, rf_pred)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    
    print(f"\nRandom Forest - Accuracy: {rf_accuracy:.4f}")
    print(f"XGBoost - Accuracy: {xgb_accuracy:.4f}")
    
    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, rf_pred))
    
    print("\nXGBoost Classification Report:")
    print(classification_report(y_test, xgb_pred))
    
    return {
        'rf_model': rf_model,
        'xgb_model': xgb_model,
        'rf_accuracy': rf_accuracy,
        'xgb_accuracy': xgb_accuracy,
        'test_data': (X_test, y_test),
        'predictions': {
            'rf': rf_pred, 
            'xgb': xgb_pred,
            'rf_proba': rf_proba,
            'xgb_proba': xgb_proba
        }
    }


def plot_feature_importance(model, feature_names, model_name, save_path='models/'):
    """Plot and save feature importance"""
    try:
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Feature Importance - {model_name}')
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{model_name}_feature_importance.png'))
        plt.close()
        print(f"✓ Saved feature importance plot: {model_name}")
    except Exception as e:
        print(f"⚠ Warning: Could not create plot for {model_name}: {e}")


def save_models(recovery_results, setback_results, encoders, scaler, save_path='models/'):
    """Save trained models and preprocessing objects"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"✓ Created directory: {save_path}")
    
    # Save recovery models
    print("Saving models...")
    joblib.dump(recovery_results['rf_model'], 
                os.path.join(save_path, 'rf_recovery_model.pkl'))
    print("  ✓ Random Forest recovery model")
    
    joblib.dump(recovery_results['xgb_model'], 
                os.path.join(save_path, 'xgb_recovery_model.pkl'))
    print("  ✓ XGBoost recovery model")
    
    # Save setback models
    joblib.dump(setback_results['rf_model'], 
                os.path.join(save_path, 'rf_setback_model.pkl'))
    print("  ✓ Random Forest setback model")
    
    joblib.dump(setback_results['xgb_model'], 
                os.path.join(save_path, 'xgb_setback_model.pkl'))
    print("  ✓ XGBoost setback model")
    
    # Save preprocessing objects
    joblib.dump(encoders, os.path.join(save_path, 'encoders.pkl'))
    print("  ✓ Encoders")
    
    joblib.dump(scaler, os.path.join(save_path, 'scaler.pkl'))
    print("  ✓ Scaler")
    
    # Save metrics
    metrics = {
        'recovery': {
            'rf_rmse': recovery_results['rf_rmse'],
            'rf_r2': recovery_results['rf_r2'],
            'xgb_rmse': recovery_results['xgb_rmse'],
            'xgb_r2': recovery_results['xgb_r2']
        },
        'setback': {
            'rf_accuracy': setback_results['rf_accuracy'],
            'xgb_accuracy': setback_results['xgb_accuracy']
        }
    }
    joblib.dump(metrics, os.path.join(save_path, 'model_metrics.pkl'))
    print("  ✓ Model metrics")
    
    print(f"\n All models saved to {save_path}")


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("INJURY RECOVERY PREDICTION - MODEL TRAINING")
    print("=" * 60)
    
    try:
        # Load data
        print("\nLoading injury data...")
        df = load_data('data/injury_data.csv')
        
        if df is None:
            print(" Failed to load data. Exiting.")
            sys.exit(1)
        
        print(f"✓ Loaded {len(df)} injury records")
        
        # Preprocess data
        print("\nPreprocessing data...")
        X, y_recovery, y_setback, encoders, scaler = preprocess_data(df)
        print(f"✓ Preprocessed data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Train recovery prediction models
        print("\n" + "=" * 60)
        print("TRAINING RECOVERY TIME PREDICTION MODELS")
        print("=" * 60)
        recovery_results = train_recovery_models(X, y_recovery)
        
        # Train setback prediction models
        print("\n" + "=" * 60)
        print("TRAINING SETBACK RISK PREDICTION MODELS")
        print("=" * 60)
        setback_results = train_setback_models(X, y_setback)
        
        # Plot feature importance (optional - won't fail if plotting fails)
        print("\nGenerating feature importance plots...")
        feature_names = X.columns.tolist()
        plot_feature_importance(recovery_results['rf_model'], feature_names, 
                              'RF_Recovery')
        plot_feature_importance(recovery_results['xgb_model'], feature_names, 
                              'XGB_Recovery')
        plot_feature_importance(setback_results['rf_model'], feature_names, 
                              'RF_Setback')
        plot_feature_importance(setback_results['xgb_model'], feature_names, 
                              'XGB_Setback')
        
        # Save models
        save_models(recovery_results, setback_results, encoders, scaler)
        
        print("\n" + "=" * 60)
        print(" TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        sys.exit(0)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(" ERROR DURING TRAINING")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
