"""
Data preprocessing utilities for injury recovery prediction
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(file_path):
    """Load injury data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def preprocess_data(df):
    """
    Preprocess injury data for machine learning models
    
    Args:
        df: pandas DataFrame with injury data
        
    Returns:
        X: Feature matrix
        y_recovery: Target variable for recovery days
        y_setback: Target variable for setback risk
        encoders: Dictionary of label encoders
        scaler: Standard scaler object
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize encoders
    encoders = {}
    
    # Encode categorical variables
    categorical_cols = ['position', 'injury_type', 'injury_severity']
    for col in categorical_cols:
        le = LabelEncoder()
        data[col + '_encoded'] = le.fit_transform(data[col])
        encoders[col] = le
    
    # Select features for model
    feature_cols = [
        'age', 'position_encoded', 'injury_type_encoded', 
        'injury_severity_encoded', 'previous_injuries', 'fitness_level'
    ]
    
    X = data[feature_cols]
    y_recovery = data['recovery_days']
    y_setback = data['had_setback']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    
    return X_scaled, y_recovery, y_setback, encoders, scaler


def encode_new_input(input_data, encoders):
    """
    Encode new input data using trained encoders
    
    Args:
        input_data: Dictionary with player injury information
        encoders: Dictionary of label encoders
        
    Returns:
        Encoded feature array
    """
    encoded_data = {
        'age': input_data['age'],
        'position_encoded': encoders['position'].transform([input_data['position']])[0],
        'injury_type_encoded': encoders['injury_type'].transform([input_data['injury_type']])[0],
        'injury_severity_encoded': encoders['injury_severity'].transform([input_data['injury_severity']])[0],
        'previous_injuries': input_data['previous_injuries'],
        'fitness_level': input_data['fitness_level']
    }
    
    return list(encoded_data.values())


def get_injury_statistics(df):
    """
    Calculate statistics for injury data
    
    Args:
        df: pandas DataFrame with injury data
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_injuries': len(df),
        'avg_recovery_days': df['recovery_days'].mean(),
        'setback_rate': (df['had_setback'].sum() / len(df)) * 100,
        'common_injuries': df['injury_type'].value_counts().to_dict(),
        'avg_recovery_by_severity': df.groupby('injury_severity')['recovery_days'].mean().to_dict(),
        'avg_recovery_by_position': df.groupby('position')['recovery_days'].mean().to_dict()
    }
    
    return stats
