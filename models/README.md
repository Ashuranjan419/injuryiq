# Models Directory

This directory will contain the trained machine learning models after running `train_model.py`.

## Generated Files

After training, the following files will be created:

- `rf_recovery_model.pkl` - Random Forest model for recovery time prediction
- `xgb_recovery_model.pkl` - XGBoost model for recovery time prediction
- `rf_setback_model.pkl` - Random Forest model for setback risk prediction
- `xgb_setback_model.pkl` - XGBoost model for setback risk prediction
- `encoders.pkl` - Label encoders for categorical variables
- `scaler.pkl` - Standard scaler for feature normalization
- `model_metrics.pkl` - Model performance metrics
- `*_feature_importance.png` - Feature importance visualizations

## Training Models

To generate these files, run:

```bash
python train_model.py
```

**Note:** These model files are required before running the Flask application.
