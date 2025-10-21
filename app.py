"""
Flask API for InjuryIQ - AI-Powered Injury Recovery Predictions
"""
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
from bson.errors import InvalidId
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import joblib
import os
import pandas as pd
import numpy as np
from datetime import datetime
from utils.data_preprocessing import encode_new_input, get_injury_statistics, load_data
from models_mongo import User, TrackedPlayer

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'fallback-secret-key-for-development')

# MongoDB Configuration - Load from environment variable
app.config['MONGO_URI'] = os.getenv('MONGO_URI', 'mongodb://localhost:27017/injuryiq')

# Initialize extensions
mongo = PyMongo(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# Load models and preprocessing objects
MODEL_PATH = 'models/'
models = {}
encoders = None
scaler = None


@login_manager.user_loader
def load_user(user_id):
    """Load user by ID for Flask-Login"""
    try:
        user_doc = mongo.db.users.find_one({'_id': ObjectId(user_id)})
        if user_doc:
            return User(user_doc)
    except InvalidId:
        # Handle invalid ObjectId (e.g., old session from SQLite migration)
        # This is expected when migrating from SQLite to MongoDB
        print(f"Invalid ObjectId format: {user_id}. Likely an old session cookie.")
    except Exception as e:
        # Catch any other unexpected errors
        print(f"Error loading user: {e}")
    return None


def load_models():
    """Load all trained models and preprocessing objects"""
    global models, encoders, scaler
    
    try:
        # Check if model directory exists
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model directory '{MODEL_PATH}' not found")
            return False
        
        # Check if all required files exist
        required_files = [
            'rf_recovery_model.pkl',
            'xgb_recovery_model.pkl',
            'rf_setback_model.pkl',
            'xgb_setback_model.pkl',
            'encoders.pkl',
            'scaler.pkl'
        ]
        
        missing_files = []
        for file in required_files:
            file_path = os.path.join(MODEL_PATH, file)
            if not os.path.exists(file_path):
                missing_files.append(file)
        
        if missing_files:
            print(f"Error: Missing model files: {', '.join(missing_files)}")
            return False
        
        # Load models
        print("Loading models...")
        models['rf_recovery'] = joblib.load(os.path.join(MODEL_PATH, 'rf_recovery_model.pkl'))
        print("✓ Random Forest recovery model loaded")
        
        models['xgb_recovery'] = joblib.load(os.path.join(MODEL_PATH, 'xgb_recovery_model.pkl'))
        print("✓ XGBoost recovery model loaded")
        
        models['rf_setback'] = joblib.load(os.path.join(MODEL_PATH, 'rf_setback_model.pkl'))
        print("✓ Random Forest setback model loaded")
        
        models['xgb_setback'] = joblib.load(os.path.join(MODEL_PATH, 'xgb_setback_model.pkl'))
        print("✓ XGBoost setback model loaded")
        
        encoders = joblib.load(os.path.join(MODEL_PATH, 'encoders.pkl'))
        print("✓ Encoders loaded")
        
        scaler = joblib.load(os.path.join(MODEL_PATH, 'scaler.pkl'))
        print("✓ Scaler loaded")
        
        # Validate loaded objects
        if encoders is None or not isinstance(encoders, dict):
            print("Error: Encoders not properly loaded")
            return False
        
        required_encoders = ['position', 'injury_type', 'injury_severity']
        for encoder_name in required_encoders:
            if encoder_name not in encoders:
                print(f"Error: Missing encoder: {encoder_name}")
                return False
        
        print("✅ All models loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        print("\nPlease train models first by running: python train_model.py")
        return False


# Authentication Routes
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """User registration"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        confirm_password = data.get('confirm_password', '')
        
        # Validation
        errors = []
        if not username or len(username) < 3:
            errors.append('Username must be at least 3 characters')
        if not email or '@' not in email:
            errors.append('Valid email is required')
        if not password or len(password) < 6:
            errors.append('Password must be at least 6 characters')
        if password != confirm_password:
            errors.append('Passwords do not match')
        
        # Check if user already exists in MongoDB
        if mongo.db.users.find_one({'username': username}):
            errors.append('Username already exists')
        if mongo.db.users.find_one({'email': email}):
            errors.append('Email already registered')
        
        if errors:
            if request.is_json:
                return jsonify({'success': False, 'errors': errors}), 400
            for error in errors:
                flash(error, 'error')
            return render_template('signup.html')
        
        # Create new user
        try:
            user_doc = User.create_user(username, email, password)
            result = mongo.db.users.insert_one(user_doc)
            
            # Create User object for login
            user_doc['_id'] = result.inserted_id
            user = User(user_doc)
            
            # Log the user in
            login_user(user)
            
            if request.is_json:
                return jsonify({'success': True, 'message': 'Account created successfully!', 'redirect': url_for('index')})
            
            flash('Account created successfully! Welcome to InjuryIQ.', 'success')
            return redirect(url_for('index'))
        except Exception as e:
            print(f"Signup error: {e}")
            error_msg = 'An error occurred. Please try again.'
            if request.is_json:
                return jsonify({'success': False, 'errors': [error_msg]}), 500
            flash(error_msg, 'error')
            return render_template('signup.html')
    
    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        
        username = data.get('username', '').strip()
        password = data.get('password', '')
        remember = data.get('remember', False)
        
        if not username or not password:
            error_msg = 'Username and password are required'
            if request.is_json:
                return jsonify({'success': False, 'error': error_msg}), 400
            flash(error_msg, 'error')
            return render_template('login.html')
        
        # Find user in MongoDB
        user_doc = mongo.db.users.find_one({'username': username})
        
        if user_doc and User.check_password(user_doc['password_hash'], password):
            user = User(user_doc)
            login_user(user, remember=remember)
            
            if request.is_json:
                return jsonify({'success': True, 'message': 'Login successful!', 'redirect': url_for('index')})
            
            flash(f'Welcome back, {user.username}!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            error_msg = 'Invalid username or password'
            if request.is_json:
                return jsonify({'success': False, 'error': error_msg}), 401
            flash(error_msg, 'error')
            return render_template('login.html')
    
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/predict')
def predict_page():
    """Prediction form page"""
    return render_template('predict.html')


@app.route('/dashboard')
def dashboard():
    """Dashboard page with statistics"""
    try:
        df = load_data('data/injury_data.csv')
        if df is not None:
            stats = get_injury_statistics(df)
            return render_template('dashboard.html', stats=stats)
        else:
            return render_template('dashboard.html', stats=None, error="Failed to load data")
    except Exception as e:
        return render_template('dashboard.html', stats=None, error=str(e))


@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for injury recovery prediction"""
    try:
        # Check if models are loaded
        if not models or encoders is None or scaler is None:
            return jsonify({
                'error': 'Models not loaded. Please contact administrator.',
                'success': False
            }), 503
        
        # Get input data
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['age', 'position', 'injury_type', 'injury_severity', 
                          'previous_injuries', 'fitness_level', 'model_type']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Prepare input data
        input_data = {
            'age': int(data['age']),
            'position': data['position'],
            'injury_type': data['injury_type'],
            'injury_severity': data['injury_severity'],
            'previous_injuries': int(data['previous_injuries']),
            'fitness_level': int(data['fitness_level'])
        }
        
        # Encode input
        encoded_input = encode_new_input(input_data, encoders)
        
        # Scale input
        scaled_input = scaler.transform([encoded_input])
        
        # Select model
        model_type = data['model_type'].lower()
        
        # Predict recovery time
        if model_type == 'random_forest':
            recovery_days = models['rf_recovery'].predict(scaled_input)[0]
            setback_prob = models['rf_setback'].predict_proba(scaled_input)[0][1]
        elif model_type == 'xgboost':
            recovery_days = models['xgb_recovery'].predict(scaled_input)[0]
            setback_prob = models['xgb_setback'].predict_proba(scaled_input)[0][1]
        else:
            return jsonify({'error': 'Invalid model type'}), 400
        
        # Determine setback risk level
        if setback_prob < 0.3:
            risk_level = 'Low'
            risk_color = 'success'
        elif setback_prob < 0.6:
            risk_level = 'Medium'
            risk_color = 'warning'
        else:
            risk_level = 'High'
            risk_color = 'danger'
        
        # Prepare response
        response = {
            'success': True,
            'player_name': data.get('player_name', 'Unknown Player'),
            'recovery_days': round(recovery_days, 1),
            'recovery_weeks': round(recovery_days / 7, 1),
            'setback_probability': round(setback_prob * 100, 1),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'model_used': model_type.replace('_', ' ').title(),
            'input_data': input_data
        }
        
        # Save to tracking if user is logged in and save_tracking is requested
        if current_user.is_authenticated and data.get('save_tracking', False):
            try:
                player_data = {
                    'player_name': data.get('player_name', 'Unknown Player'),
                    'age': input_data['age'],
                    'position': input_data['position'],
                    'injury_type': input_data['injury_type'],
                    'injury_severity': input_data['injury_severity'],
                    'previous_injuries': input_data['previous_injuries'],
                    'fitness_level': input_data['fitness_level'],
                    'predicted_recovery_days': recovery_days,
                    'predicted_setback_risk': risk_level,
                    'setback_probability': setback_prob,
                    'notes': data.get('notes', '')
                }
                
                tracked_player_doc = TrackedPlayer.create_tracked_player(current_user.id, player_data)
                result = mongo.db.tracked_players.insert_one(tracked_player_doc)
                
                response['tracking_saved'] = True
                response['tracking_id'] = str(result.inserted_id)
            except Exception as e:
                print(f"Error saving tracking: {e}")
                response['tracking_saved'] = False
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """API endpoint for injury statistics"""
    try:
        df = load_data('data/injury_data.csv')
        if df is not None:
            stats = get_injury_statistics(df)
            return jsonify({'success': True, 'statistics': stats}), 200
        else:
            return jsonify({'error': 'Failed to load data', 'success': False}), 500
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/injury-types', methods=['GET'])
def get_injury_types():
    """Get list of available injury types"""
    try:
        injury_types = list(encoders['injury_type'].classes_)
        return jsonify({'success': True, 'injury_types': injury_types}), 200
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


# Tracking Routes
@app.route('/tracking')
@login_required
def tracking():
    """Player tracking page"""
    return render_template('tracking.html')


@app.route('/api/tracking', methods=['GET'])
@login_required
def get_tracked_players():
    """Get all tracked players for current user"""
    try:
        # Get all tracked players for the current user from MongoDB
        players_cursor = mongo.db.tracked_players.find({'user_id': current_user.id}).sort('prediction_date', -1)
        
        players_data = []
        for player_doc in players_cursor:
            players_data.append(TrackedPlayer.format_player_for_api(player_doc))
        
        return jsonify({'success': True, 'players': players_data}), 200
        
    except Exception as e:
        print(f"Error fetching tracked players: {e}")
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/tracking/<player_id>', methods=['DELETE'])
@login_required
def delete_tracked_player(player_id):
    """Delete a tracked player"""
    try:
        # Delete from MongoDB
        result = mongo.db.tracked_players.delete_one({
            '_id': ObjectId(player_id),
            'user_id': current_user.id
        })
        
        if result.deleted_count == 0:
            return jsonify({'error': 'Player not found', 'success': False}), 404
        
        return jsonify({'success': True, 'message': 'Player removed from tracking'}), 200
        
    except Exception as e:
        print(f"Error deleting player: {e}")
        return jsonify({'error': str(e), 'success': False}), 500
    """Delete a tracked player"""
    try:
        player = TrackedPlayer.query.filter_by(id=player_id, user_id=current_user.id).first()
        
        if not player:
            return jsonify({'error': 'Player not found', 'success': False}), 404
        
        db.session.delete(player)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Player removed from tracking'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/tracking/<player_id>/recover', methods=['PUT'])
@login_required
def mark_recovered(player_id):
    """Mark a player as recovered"""
    try:
        # Update in MongoDB
        result = mongo.db.tracked_players.update_one(
            {'_id': ObjectId(player_id), 'user_id': current_user.id},
            {'$set': {'is_recovered': True}}
        )
        
        if result.matched_count == 0:
            return jsonify({'error': 'Player not found', 'success': False}), 404
        
        return jsonify({'success': True, 'message': 'Player marked as recovered'}), 200
        
    except Exception as e:
        print(f"Error marking player as recovered: {e}")
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/tracking/<player_id>/notes', methods=['PUT'])
@login_required
def update_notes(player_id):
    """Update notes for a tracked player"""
    try:
        data = request.get_json()
        
        # Update in MongoDB
        result = mongo.db.tracked_players.update_one(
            {'_id': ObjectId(player_id), 'user_id': current_user.id},
            {'$set': {'notes': data.get('notes', '')}}
        )
        
        if result.matched_count == 0:
            return jsonify({'error': 'Player not found', 'success': False}), 404
        
        return jsonify({'success': True, 'message': 'Notes updated'}), 200
        
    except Exception as e:
        print(f"Error updating notes: {e}")
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/positions', methods=['GET'])
def get_positions():
    """Get list of available positions"""
    try:
        if encoders is None or 'position' not in encoders:
            return jsonify({'error': 'Encoders not loaded', 'success': False}), 503
        positions = list(encoders['position'].classes_)
        return jsonify({'success': True, 'positions': positions}), 200
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify system status"""
    status = {
        'status': 'healthy',
        'models_loaded': False,
        'encoders_loaded': False,
        'scaler_loaded': False,
        'database_connected': False,
        'issues': [],
        'environment': os.environ.get('RENDER', 'local'),
        'model_path_exists': os.path.exists(MODEL_PATH),
        'model_files': []
    }
    
    # Check if model directory exists and list files
    if os.path.exists(MODEL_PATH):
        try:
            status['model_files'] = os.listdir(MODEL_PATH)
        except Exception as e:
            status['issues'].append(f'Cannot list model directory: {str(e)}')
    else:
        status['issues'].append(f'Model directory does not exist: {MODEL_PATH}')
    
    # Check models
    required_models = ['rf_recovery', 'xgb_recovery', 'rf_setback', 'xgb_setback']
    if models and all(model_name in models for model_name in required_models):
        status['models_loaded'] = True
    else:
        status['issues'].append('Models not loaded')
        missing = [m for m in required_models if m not in models]
        if missing:
            status['missing_models'] = missing
    
    # Check encoders
    if encoders is not None and isinstance(encoders, dict):
        required_encoders = ['position', 'injury_type', 'injury_severity']
        if all(enc in encoders for enc in required_encoders):
            status['encoders_loaded'] = True
            status['available_positions'] = list(encoders['position'].classes_)
            status['available_injury_types'] = list(encoders['injury_type'].classes_)
            status['available_severities'] = list(encoders['injury_severity'].classes_)
        else:
            status['issues'].append('Encoders incomplete')
    else:
        status['issues'].append('Encoders not loaded')
    
    # Check scaler
    if scaler is not None:
        status['scaler_loaded'] = True
    else:
        status['issues'].append('Scaler not loaded')
    
    # Check database
    try:
        mongo.db.command('ping')
        status['database_connected'] = True
    except Exception as e:
        status['issues'].append(f'Database error: {str(e)}')
    
    # Check if data file exists
    data_file = 'data/injury_data.csv'
    status['data_file_exists'] = os.path.exists(data_file)
    if not status['data_file_exists']:
        status['issues'].append(f'Data file not found: {data_file}')
    
    # Determine overall status
    if status['issues']:
        status['status'] = 'unhealthy'
        return jsonify(status), 503
    
    return jsonify(status), 200


@app.route('/api/admin/train-models', methods=['POST'])
def admin_train_models():
    """Admin endpoint to manually trigger model training"""
    try:
        print("Starting manual model training...")
        import subprocess
        result = subprocess.run(['python', 'train_model.py'], 
                              capture_output=True, 
                              text=True, 
                              timeout=600)
        
        if result.returncode == 0:
            # Try to reload models
            global models, encoders, scaler
            models = {}
            if load_models():
                return jsonify({
                    'success': True,
                    'message': 'Models trained and loaded successfully',
                    'output': result.stdout
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'message': 'Training completed but models failed to load',
                    'output': result.stdout,
                    'error': result.stderr
                }), 500
        else:
            return jsonify({
                'success': False,
                'message': 'Model training failed',
                'output': result.stdout,
                'error': result.stderr
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Training error: {str(e)}'
        }), 500


@app.errorhandler(404)
def not_found(error):
    """404 error handler"""
    try:
        return render_template('404.html'), 404
    except:
        return jsonify({'error': 'Page not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """500 error handler"""
    try:
        return render_template('500.html'), 500
    except:
        return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Load models
    models_loaded = load_models()
    
    if not models_loaded:
        print("\n" + "=" * 60)
        print("⚠️  WARNING: Models not found!")
        print("=" * 60 + "\n")
        
        # On Render or production, try to train models automatically
        is_production = os.environ.get('RENDER', False) or os.environ.get('PRODUCTION', False)
        if is_production:
            print("🔄 Attempting to train models automatically...")
            try:
                import subprocess
                result = subprocess.run(['python', 'train_model.py'], 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=600)  # 10 minute timeout
                print(result.stdout)
                if result.returncode == 0:
                    print("✅ Model training completed!")
                    # Try loading models again
                    models_loaded = load_models()
                    if not models_loaded:
                        print("❌ Models still not loading after training")
                else:
                    print(f"❌ Model training failed with code {result.returncode}")
                    print(result.stderr)
            except Exception as e:
                print(f"❌ Failed to auto-train models: {e}")
        
        if not models_loaded:
            print("Please train models first by running: python train_model.py")
            print("The server will start but prediction endpoints will not work.")
            print("=" * 60 + "\n")
            
            if not is_production:
                exit(1)
    
    # Test MongoDB connection
    try:
        with app.app_context():
            mongo.db.command('ping')
            print("✅ MongoDB connected successfully")
            
            # Create indexes for better performance
            mongo.db.users.create_index('username', unique=True)
            mongo.db.users.create_index('email', unique=True)
            mongo.db.tracked_players.create_index('user_id')
            print("✅ Database indexes created")
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ MongoDB Connection Error!")
        print(f"Error: {e}")
        print("\nPlease ensure:")
        print("1. MongoDB is running")
        print("2. Update MONGO_URI in app.py with your MongoDB connection string")
        print("=" * 60 + "\n")
        exit(1)
    
    # Get configuration from environment
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    print("\n" + "=" * 60)
    print("🧠 INJURYIQ - AI-POWERED INJURY PREDICTIONS")
    print("=" * 60)
    print(f"Server starting on http://127.0.0.1:{port}")
    print(f"Debug mode: {debug}")
    print("Press CTRL+C to quit")
    print("=" * 60 + "\n")
    
    app.run(debug=debug, host='0.0.0.0', port=port)
