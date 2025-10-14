"""
MongoDB models for InjuryIQ
Handles user authentication and player tracking with MongoDB
"""

from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from bson.objectid import ObjectId


class User(UserMixin):
    """User model for authentication with MongoDB"""
    
    def __init__(self, user_data):
        """Initialize user from MongoDB document"""
        self.id = str(user_data.get('_id', ''))
        self.username = user_data.get('username', '')
        self.email = user_data.get('email', '')
        self.password_hash = user_data.get('password_hash', '')
        self.created_at = user_data.get('created_at', datetime.utcnow())
    
    def get_id(self):
        """Return user ID as string for Flask-Login"""
        return self.id
    
    @staticmethod
    def create_user(username, email, password):
        """Create a new user document"""
        return {
            'username': username,
            'email': email,
            'password_hash': generate_password_hash(password),
            'created_at': datetime.utcnow()
        }
    
    @staticmethod
    def check_password(password_hash, password):
        """Verify password against hash"""
        return check_password_hash(password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'


class TrackedPlayer:
    """Model for tracking player injury recovery with MongoDB"""
    
    @staticmethod
    def create_tracked_player(user_id, player_data):
        """Create a new tracked player document"""
        prediction_date = datetime.utcnow()
        recovery_days = player_data['predicted_recovery_days']
        expected_recovery_date = prediction_date + timedelta(days=recovery_days)
        
        return {
            'user_id': user_id,
            'player_name': player_data['player_name'],
            'age': player_data['age'],
            'position': player_data['position'],
            'injury_type': player_data['injury_type'],
            'injury_severity': player_data['injury_severity'],
            'previous_injuries': player_data['previous_injuries'],
            'fitness_level': player_data['fitness_level'],
            'predicted_recovery_days': recovery_days,
            'predicted_setback_risk': player_data['predicted_setback_risk'],
            'setback_probability': player_data['setback_probability'],
            'prediction_date': prediction_date,
            'expected_recovery_date': expected_recovery_date,
            'is_recovered': False,
            'notes': player_data.get('notes', '')
        }
    
    @staticmethod
    def calculate_days_remaining(expected_recovery_date, is_recovered=False):
        """Calculate days remaining in recovery"""
        if is_recovered:
            return 0
        
        today = datetime.utcnow()
        if today >= expected_recovery_date:
            return 0
        
        delta = expected_recovery_date - today
        return delta.days
    
    @staticmethod
    def calculate_recovery_progress(prediction_date, predicted_recovery_days):
        """Calculate recovery progress percentage (0-100)"""
        days_passed = (datetime.utcnow() - prediction_date).days
        
        if days_passed >= predicted_recovery_days:
            return 100
        
        progress = (days_passed / predicted_recovery_days) * 100
        return round(progress, 1)
    
    @staticmethod
    def get_status(is_recovered, days_remaining, recovery_progress):
        """Get current recovery status"""
        if is_recovered:
            return 'Recovered'
        elif days_remaining == 0:
            return 'Recovery Complete'
        elif days_remaining <= 3:
            return 'Almost Ready'
        elif recovery_progress >= 75:
            return 'Final Phase'
        elif recovery_progress >= 50:
            return 'Progressing Well'
        else:
            return 'Early Recovery'
    
    @staticmethod
    def format_player_for_api(player_doc):
        """Format MongoDB document for API response"""
        days_remaining = TrackedPlayer.calculate_days_remaining(
            player_doc['expected_recovery_date'],
            player_doc.get('is_recovered', False)
        )
        recovery_progress = TrackedPlayer.calculate_recovery_progress(
            player_doc['prediction_date'],
            player_doc['predicted_recovery_days']
        )
        status = TrackedPlayer.get_status(
            player_doc.get('is_recovered', False),
            days_remaining,
            recovery_progress
        )
        
        return {
            'id': str(player_doc['_id']),
            'player_name': player_doc['player_name'],
            'age': player_doc['age'],
            'position': player_doc['position'],
            'injury_type': player_doc['injury_type'],
            'injury_severity': player_doc['injury_severity'],
            'previous_injuries': player_doc['previous_injuries'],
            'fitness_level': player_doc['fitness_level'],
            'predicted_recovery_days': round(player_doc['predicted_recovery_days'], 1),
            'predicted_setback_risk': player_doc['predicted_setback_risk'],
            'setback_probability': round(player_doc['setback_probability'] * 100, 1),
            'prediction_date': player_doc['prediction_date'].strftime('%Y-%m-%d %H:%M'),
            'expected_recovery_date': player_doc['expected_recovery_date'].strftime('%Y-%m-%d'),
            'days_remaining': days_remaining,
            'recovery_progress': recovery_progress,
            'status': status,
            'is_recovered': player_doc.get('is_recovered', False),
            'notes': player_doc.get('notes', '')
        }
