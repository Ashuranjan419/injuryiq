"""
Quick test script to verify models work correctly
"""
import joblib
import os

def test_models():
    """Test if models can be loaded and used"""
    print("Testing model loading...")
    
    MODEL_PATH = 'models/'
    
    try:
        # Load models
        print("Loading encoders...")
        encoders = joblib.load(os.path.join(MODEL_PATH, 'encoders.pkl'))
        print(f"✓ Encoders loaded: {list(encoders.keys())}")
        
        print("\nLoading scaler...")
        scaler = joblib.load(os.path.join(MODEL_PATH, 'scaler.pkl'))
        print("✓ Scaler loaded")
        
        print("\nLoading models...")
        rf_recovery = joblib.load(os.path.join(MODEL_PATH, 'rf_recovery_model.pkl'))
        print("✓ Random Forest recovery model loaded")
        
        xgb_recovery = joblib.load(os.path.join(MODEL_PATH, 'xgb_recovery_model.pkl'))
        print("✓ XGBoost recovery model loaded")
        
        rf_setback = joblib.load(os.path.join(MODEL_PATH, 'rf_setback_model.pkl'))
        print("✓ Random Forest setback model loaded")
        
        xgb_setback = joblib.load(os.path.join(MODEL_PATH, 'xgb_setback_model.pkl'))
        print("✓ XGBoost setback model loaded")
        
        # Check encoder classes
        print("\n" + "="*50)
        print("Available Classes:")
        print("="*50)
        print(f"Positions: {list(encoders['position'].classes_)}")
        print(f"Injury Types: {list(encoders['injury_type'].classes_)}")
        print(f"Severities: {list(encoders['injury_severity'].classes_)}")
        
        # Test prediction
        print("\n" + "="*50)
        print("Testing Sample Prediction:")
        print("="*50)
        
        test_input = {
            'age': 25,
            'position': 'Forward',
            'injury_type': 'Hamstring Strain',
            'injury_severity': 'Moderate',
            'previous_injuries': 1,
            'fitness_level': 8
        }
        
        print(f"Input: {test_input}")
        
        # Encode
        encoded = [
            test_input['age'],
            encoders['position'].transform([test_input['position']])[0],
            encoders['injury_type'].transform([test_input['injury_type']])[0],
            encoders['injury_severity'].transform([test_input['injury_severity']])[0],
            test_input['previous_injuries'],
            test_input['fitness_level']
        ]
        
        # Scale
        scaled = scaler.transform([encoded])
        
        # Predict
        recovery_days = rf_recovery.predict(scaled)[0]
        setback_prob = rf_setback.predict_proba(scaled)[0][1]
        
        print(f"\n✅ Prediction successful!")
        print(f"Recovery Days: {recovery_days:.1f}")
        print(f"Setback Probability: {setback_prob*100:.1f}%")
        
        print("\n" + "="*50)
        print("✅ ALL TESTS PASSED!")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_models()
    exit(0 if success else 1)
