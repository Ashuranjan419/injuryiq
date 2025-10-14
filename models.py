"""
Database models for InjuryIQ
Handles user authentication and player tracking
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """User model for authentication"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship to tracked players
    tracked_players = db.relationship('TrackedPlayer', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Verify password against hash"""
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'


class TrackedPlayer(db.Model):
    """Model for tracking player injury recovery"""
    __tablename__ = 'tracked_players'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    
    # Player information
    player_name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    position = db.Column(db.String(50), nullable=False)
    
    # Injury details
    injury_type = db.Column(db.String(100), nullable=False)
    injury_severity = db.Column(db.String(50), nullable=False)
    previous_injuries = db.Column(db.Integer, nullable=False)
    fitness_level = db.Column(db.String(50), nullable=False)
    
    # Prediction results
    predicted_recovery_days = db.Column(db.Float, nullable=False)
    predicted_setback_risk = db.Column(db.String(50), nullable=False)
    setback_probability = db.Column(db.Float, nullable=False)
    
    # Tracking information
    prediction_date = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    expected_recovery_date = db.Column(db.DateTime, nullable=False)
    is_recovered = db.Column(db.Boolean, default=False)
    notes = db.Column(db.Text)
    
    def __init__(self, **kwargs):
        super(TrackedPlayer, self).__init__(**kwargs)
        # Calculate expected recovery date
        if self.predicted_recovery_days and self.prediction_date:
            self.expected_recovery_date = self.prediction_date + timedelta(days=self.predicted_recovery_days)
    
    @property
    def days_remaining(self):
        """Calculate days remaining in recovery"""
        if self.is_recovered:
            return 0
        
        today = datetime.utcnow()
        if today >= self.expected_recovery_date:
            return 0
        
        delta = self.expected_recovery_date - today
        return delta.days
    
    @property
    def recovery_progress(self):
        """Calculate recovery progress percentage (0-100)"""
        total_days = self.predicted_recovery_days
        days_passed = (datetime.utcnow() - self.prediction_date).days
        
        if days_passed >= total_days:
            return 100
        
        progress = (days_passed / total_days) * 100
        return round(progress, 1)
    
    @property
    def status(self):
        """Get current recovery status"""
        if self.is_recovered:
            return 'Recovered'
        elif self.days_remaining == 0:
            return 'Recovery Complete'
        elif self.days_remaining <= 3:
            return 'Almost Ready'
        elif self.recovery_progress >= 75:
            return 'Final Phase'
        elif self.recovery_progress >= 50:
            return 'Progressing Well'
        else:
            return 'Early Recovery'
    
    def __repr__(self):
        return f'<TrackedPlayer {self.player_name} - {self.injury_type}>'
