# Quick Start Guide

## 🚀 Getting Started with InjuryIQ

### Prerequisites Check ✅
- [x] Python 3.8+ installed
- [x] Virtual environment created
- [x] Dependencies installed
- [x] ML models trained

### Ready to Launch! 🎉

You have two options to run the application:

## Option 1: Using VS Code Tasks (Recommended)

### Train Models (Already Done ✅)
1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS)
2. Type "Run Task"
3. Select "Train ML Models"

### Run Flask Server
1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS)
2. Type "Run Task"
3. Select "Run Flask Server"
4. Open browser to `http://127.0.0.1:5000`

## Option 2: Using Terminal

### Train Models (Already Done ✅)
```bash
C:/Users/ranja/OneDrive/Desktop/bcnproject/.venv/Scripts/python.exe train_model.py
```

### Run Flask Server
```bash
C:/Users/ranja/OneDrive/Desktop/bcnproject/.venv/Scripts/python.exe app.py
```

Then open your browser to: **http://127.0.0.1:5000**

## 📱 Using the Application

### 1. Home Page
- Overview of features
- Navigate to Predict or Dashboard

### 2. Make a Prediction
- Go to **Predict** page
- Fill in player details:
  - Player Name
  - Age (18-45)
  - Position
  - Injury Type
  - Severity
  - Previous Injuries
  - Fitness Level (1-10)
- Choose model (Random Forest or XGBoost)
- Click **Predict Recovery Time**
- View results with recovery days and setback risk

### 3. View Dashboard
- See injury statistics
- Average recovery times
- Common injury types
- Setback rates

## 🎯 Example Prediction

**Input:**
- Player: John Smith
- Age: 25
- Position: Forward
- Injury: Hamstring Strain
- Severity: Moderate
- Previous Injuries: 2
- Fitness Level: 8

**Output:**
- Recovery Time: ~21 days (3 weeks)
- Setback Risk: ~35% (Medium)

## 🛠️ Troubleshooting

### Port Already in Use?
Change port in `app.py` line 143:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change 5000 to 5001
```

### Models Not Found?
Run training:
```bash
python train_model.py
```

## 📚 Next Steps

1. ✅ Explore the prediction interface
2. ✅ Try different injury scenarios
3. ✅ View the dashboard statistics
4. ✅ Compare Random Forest vs XGBoost predictions
5. ✅ Customize the dataset in `data/injury_data.csv`
6. ✅ Retrain models with new data

## 🎉 You're All Set!

Your InjuryIQ platform is fully configured and ready to use!

**Happy Predicting! 🧠⚽🏥📊**
