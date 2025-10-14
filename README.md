# 🧠 InjuryIQ - AI-Powered Injury Recovery Predictions

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)
![ML](https://img.shields.io/badge/ML-Random%20Forest%20%7C%20XGBoost-orange.svg)
![AI](https://img.shields.io/badge/AI-Powered-purple.svg)

An intelligent AI-powered platform for predicting football player injury recovery time and setback risk using advanced Machine Learning algorithms (Random Forest and XGBoost).

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [API Endpoints](#api-endpoints)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

**InjuryIQ** is an intelligent AI-powered platform designed to help football coaches and medical staff make data-driven decisions about injured players. By analyzing various factors such as injury type, severity, player age, position, fitness level, and injury history, our advanced machine learning models provide accurate predictions for recovery time and assess the risk of setbacks during rehabilitation.

## ✨ Features

- **Dual ML Models**: Choose between Random Forest and XGBoost algorithms
- **Accurate Predictions**: Predict injury recovery time in days and weeks
- **Risk Assessment**: Calculate probability of setbacks during recovery
- **Interactive Dashboard**: View injury statistics and trends
- **Real-Time Analysis**: Get instant predictions through an intuitive web interface
- **Comprehensive Data**: Based on historical football injury data
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## 🛠️ Tech Stack

### Backend
- **Python 3.8+**
- **Flask** - Web framework
- **scikit-learn** - Random Forest implementation
- **XGBoost** - Gradient boosting framework
- **pandas & numpy** - Data manipulation
- **joblib** - Model serialization

### Frontend
- **HTML5 & CSS3**
- **JavaScript (Vanilla)**
- **Responsive Design**

### Machine Learning
- **Random Forest Regressor/Classifier**
- **XGBoost Regressor/Classifier**
- **Standard Scaling**
- **Label Encoding**

## 📁 Project Structure

```
bcnproject/
├── .github/
│   └── copilot-instructions.md
├── app.py                      # Flask application
├── train_model.py              # ML model training script
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── data/
│   └── injury_data.csv        # Training dataset
├── models/                     # Trained models (generated)
│   ├── rf_recovery_model.pkl
│   ├── xgb_recovery_model.pkl
│   ├── rf_setback_model.pkl
│   ├── xgb_setback_model.pkl
│   ├── encoders.pkl
│   ├── scaler.pkl
│   └── model_metrics.pkl
├── static/
│   ├── css/
│   │   └── style.css          # Stylesheet
│   └── js/
│       ├── main.js            # Main JavaScript
│       └── predict.js         # Prediction logic
├── templates/
│   ├── index.html             # Home page
│   ├── predict.html           # Prediction form
│   └── dashboard.html         # Statistics dashboard
└── utils/
    └── data_preprocessing.py  # Data utilities
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Clone the Repository
```bash
cd c:\Users\ranja\OneDrive\Desktop\bcnproject
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Train the Models
Before running the application, you need to train the machine learning models:

```bash
python train_model.py
```

This will:
- Load the injury dataset
- Preprocess the data
- Train Random Forest and XGBoost models
- Save trained models to the `models/` directory
- Generate feature importance plots

Expected output:
```
==============================================================
INJURY RECOVERY PREDICTION - MODEL TRAINING
==============================================================

Loading injury data...
Loaded 50 injury records

Preprocessing data...

==============================================================
TRAINING RECOVERY TIME PREDICTION MODELS
==============================================================
Training Random Forest for recovery prediction...
Training XGBoost for recovery prediction...

Random Forest - RMSE: X.XX, R²: X.XXXX
XGBoost - RMSE: X.XX, R²: X.XXXX

==============================================================
TRAINING SETBACK RISK PREDICTION MODELS
==============================================================
Training Random Forest for setback prediction...
Training XGBoost for setback prediction...

Random Forest - Accuracy: X.XXXX
XGBoost - Accuracy: X.XXXX

Models saved to models/

==============================================================
TRAINING COMPLETED SUCCESSFULLY!
==============================================================
```

### Step 5: Run the Application
```bash
python app.py
```

The application will start on `http://127.0.0.1:5000`

## 📖 Usage

### Web Interface

1. **Home Page** (`/`)
   - Overview of the system
   - Key features
   - Quick access to prediction and dashboard

2. **Prediction Page** (`/predict`)
   - Fill in player information:
     - Player Name
     - Age (18-45)
     - Position (Forward, Midfielder, Defender, Goalkeeper)
     - Injury Type
     - Injury Severity (Mild, Moderate, Severe)
     - Previous Injuries (0-20)
     - Fitness Level (1-10)
     - Model Type (Random Forest or XGBoost)
   - Submit to get predictions
   - View recovery time and setback risk

3. **Dashboard** (`/dashboard`)
   - Total injuries statistics
   - Average recovery days
   - Setback rate
   - Most common injuries
   - Recovery time by severity
   - Recovery time by position

## 🤖 Model Training

### Dataset
The system uses a dataset of 50 football player injury records with the following features:

**Input Features:**
- Age
- Position (Forward, Midfielder, Defender, Goalkeeper)
- Injury Type (Hamstring Strain, ACL Tear, Ankle Sprain, etc.)
- Injury Severity (Mild, Moderate, Severe)
- Previous Injuries (count)
- Fitness Level (1-10 scale)

**Target Variables:**
- Recovery Days (continuous)
- Had Setback (binary: 0 or 1)

### Models

**1. Recovery Time Prediction (Regression)**
- Random Forest Regressor
- XGBoost Regressor
- Metrics: RMSE, R² Score

**2. Setback Risk Prediction (Classification)**
- Random Forest Classifier
- XGBoost Classifier
- Metrics: Accuracy, Precision, Recall, F1-Score

### Feature Importance
The training script generates feature importance plots for all models, saved in the `models/` directory.

## 🔌 API Endpoints

### POST `/api/predict`
Predict injury recovery time and setback risk.

**Request Body:**
```json
{
  "player_name": "John Doe",
  "age": 25,
  "position": "Forward",
  "injury_type": "Hamstring Strain",
  "injury_severity": "Moderate",
  "previous_injuries": 2,
  "fitness_level": 8,
  "model_type": "random_forest"
}
```

**Response:**
```json
{
  "success": true,
  "player_name": "John Doe",
  "recovery_days": 21.5,
  "recovery_weeks": 3.1,
  "setback_probability": 35.2,
  "risk_level": "Medium",
  "risk_color": "warning",
  "model_used": "Random Forest"
}
```

### GET `/api/statistics`
Get injury statistics from the dataset.

**Response:**
```json
{
  "success": true,
  "statistics": {
    "total_injuries": 50,
    "avg_recovery_days": 45.2,
    "setback_rate": 42.0,
    "common_injuries": {...},
    "avg_recovery_by_severity": {...},
    "avg_recovery_by_position": {...}
  }
}
```

### GET `/api/injury-types`
Get list of available injury types.

### GET `/api/positions`
Get list of available player positions.

## 🎨 Customization

### Adding New Injury Types
1. Add data to `data/injury_data.csv`
2. Retrain models: `python train_model.py`
3. Update the dropdown in `templates/predict.html` (optional)

### Modifying Model Parameters
Edit `train_model.py` and adjust hyperparameters:

```python
# Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100,  # Number of trees
    max_depth=10,      # Maximum depth
    min_samples_split=5,
    random_state=42
)

# XGBoost
xgb_model = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
```

## 🐛 Troubleshooting

### Models Not Found Error
If you see "Models not found!" error:
```bash
python train_model.py
```

### Import Errors
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Port Already in Use
Change the port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
```

## 📊 Model Performance

The models are evaluated using:
- **Recovery Time**: RMSE (Root Mean Square Error) and R² Score
- **Setback Risk**: Accuracy, Precision, Recall, F1-Score

Performance metrics are saved in `models/model_metrics.pkl` and displayed during training.

## 🔮 Future Enhancements

- [ ] User authentication and player profiles
- [ ] Historical prediction tracking
- [ ] Advanced data visualization with charts
- [ ] Export reports to PDF
- [ ] Mobile app integration
- [ ] Real-time data updates
- [ ] Multi-language support
- [ ] Integration with wearable devices

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Ashutosh Ranjan - Initial work

## 🙏 Acknowledgments

- Inspired by sports medicine research
- Thanks to the scikit-learn and XGBoost communities
- Flask documentation and tutorials

## 📞 Contact

For questions or support, please open an issue on GitHub.

---

**Made with ❤️ for Football**
