"""
DATABASE SETUP: SQLite database with encryption for user data
Stores: Users, Chat History, Meal Logs, Progress Tracking
"""
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from cryptography.fernet import Fernet
import os
import json

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

DB_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "instance", "fitness_app.db")
)

DATABASE_URI = os.getenv(
    'DATABASE_URL',
    f"sqlite:///{DB_PATH}" 
)
if DATABASE_URI.startswith("postgres://"):
    DATABASE_URI = DATABASE_URI.replace("postgres://", "postgresql://", 1)


# Initialize database
db = SQLAlchemy()

# Encryption key (in production, store this securely in environment variable)
# For now, generate and save to file
ENCRYPTION_KEY_FILE = 'backend/encryption.key'

def get_encryption_key():
    """Get or create encryption key"""
    if os.path.exists(ENCRYPTION_KEY_FILE):
        with open(ENCRYPTION_KEY_FILE, 'rb') as f:
            return f.read()
    else:
        key = Fernet.generate_key()
        os.makedirs('backend', exist_ok=True)
        with open(ENCRYPTION_KEY_FILE, 'wb') as f:
            f.write(key)
        return key

# Create cipher
ENCRYPTION_KEY = get_encryption_key()
cipher = Fernet(ENCRYPTION_KEY)

def encrypt_data(data: str) -> str:
    """Encrypt sensitive data"""
    if data is None:
        return None
    return cipher.encrypt(data.encode()).decode()

def decrypt_data(encrypted_data: str) -> str:
    """Decrypt sensitive data"""
    if encrypted_data is None:
        return None
    return cipher.decrypt(encrypted_data.encode()).decode()

# DATABASE MODELS
class User(db.Model):
    """User account table"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)  # Hashed, not encrypted
    
    # Personal info (ENCRYPTED)
    age_encrypted = db.Column(db.String(200))
    gender_encrypted = db.Column(db.String(200))
    height_encrypted = db.Column(db.String(200))  # cm
    weight_encrypted = db.Column(db.String(200))  # kg (start weight)
    
    # Fitness profile (ENCRYPTED)
    goal_encrypted = db.Column(db.String(200))  # weight_loss, muscle_gain, maintenance
    activity_level_encrypted = db.Column(db.String(200))
    gym_days_encrypted = db.Column(db.String(200))
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    # Relationships
    chat_history = db.relationship('ChatMessage', backref='user', lazy=True, cascade='all, delete-orphan')
    meal_logs = db.relationship('MealLog', backref='user', lazy=True, cascade='all, delete-orphan')
    progress_logs = db.relationship('ProgressLog', backref='user', lazy=True, cascade='all, delete-orphan')
    current_plan = db.relationship('UserPlan', backref='user', uselist=False, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<User {self.username}>'
    
    # Helper methods for encrypted fields
    def set_age(self, age):
        self.age_encrypted = encrypt_data(str(age))
    
    def get_age(self):
        return int(decrypt_data(self.age_encrypted)) if self.age_encrypted else None
    
    def set_gender(self, gender):
        self.gender_encrypted = encrypt_data(gender)
    
    def get_gender(self):
        return decrypt_data(self.gender_encrypted) if self.gender_encrypted else None
    
    def set_height(self, height):
        self.height_encrypted = encrypt_data(str(height))
    
    def get_height(self):
        return float(decrypt_data(self.height_encrypted)) if self.height_encrypted else None
    
    def set_weight(self, weight):
        self.weight_encrypted = encrypt_data(str(weight))
    
    def get_weight(self):
        return float(decrypt_data(self.weight_encrypted)) if self.weight_encrypted else None
    
    def set_goal(self, goal):
        self.goal_encrypted = encrypt_data(goal)
    
    def get_goal(self):
        return decrypt_data(self.goal_encrypted) if self.goal_encrypted else None
    
    def set_activity_level(self, level):
        self.activity_level_encrypted = encrypt_data(level)
    
    def get_activity_level(self):
        return decrypt_data(self.activity_level_encrypted) if self.activity_level_encrypted else None
    
    def set_gym_days(self, days):
        self.gym_days_encrypted = encrypt_data(str(days))
    
    def get_gym_days(self):
        return int(decrypt_data(self.gym_days_encrypted)) if self.gym_days_encrypted else None
    
    def get_profile_dict(self):
        """Get user profile as dict for agent"""
        return {
            'age': self.get_age(),
            'weight': self.get_weight(),
            'height': self.get_height(),
            'gender': self.get_gender(),
            'goal': self.get_goal(),
            'activity_level': self.get_activity_level(),
            'gym_days': self.get_gym_days()
        }

class UserPlan(db.Model):
    """Current fitness plan for user"""
    __tablename__ = 'user_plans'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, unique=True)
    
    # Plan data (stored as encrypted JSON)
    plan_data_encrypted = db.Column(db.Text)
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def set_plan(self, plan_dict):
        """Store plan as encrypted JSON"""
        plan_json = json.dumps(plan_dict)
        self.plan_data_encrypted = encrypt_data(plan_json)
    
    def get_plan(self):
        """Retrieve plan as dict"""
        if self.plan_data_encrypted:
            plan_json = decrypt_data(self.plan_data_encrypted)
            return json.loads(plan_json)
        return None

class ChatMessage(db.Model):
    """Chat history between user and agent (ENCRYPTED)"""
    __tablename__ = 'chat_messages'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    role = db.Column(db.String(20), nullable=False)  # 'user' or 'agent'
    message_encrypted = db.Column(db.Text, nullable=False)
    
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def set_message(self, message):
        self.message_encrypted = encrypt_data(message)
    
    def get_message(self):
        return decrypt_data(self.message_encrypted)
    
    def to_dict(self):
        return {
            'role': self.role,
            'message': self.get_message(),
            'timestamp': self.timestamp.isoformat()
        }

class MealLog(db.Model):
    """Daily meal logging (ENCRYPTED)"""
    __tablename__ = 'meal_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    date = db.Column(db.Date, nullable=False)
    meal_type = db.Column(db.String(20))  # breakfast, lunch, dinner, snack
    
    # Meal data (ENCRYPTED)
    meal_name_encrypted = db.Column(db.String(200))
    calories = db.Column(db.Float)
    protein = db.Column(db.Float)
    carbs = db.Column(db.Float)
    fat = db.Column(db.Float)
    
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def set_meal_name(self, name):
        self.meal_name_encrypted = encrypt_data(name)
    
    def get_meal_name(self):
        return decrypt_data(self.meal_name_encrypted)
    
    def to_dict(self):
        return {
            'id': self.id,
            'date': self.date.isoformat(),
            'meal_type': self.meal_type,
            'meal_name': self.get_meal_name(),
            'calories': self.calories,
            'protein': self.protein,
            'carbs': self.carbs,
            'fat': self.fat
        }

class ProgressLog(db.Model):
    """Weekly progress tracking (ENCRYPTED)"""
    __tablename__ = 'progress_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    week_number = db.Column(db.Integer, nullable=False)
    date = db.Column(db.Date, nullable=False)
    
    # Progress data (ENCRYPTED)
    weight_encrypted = db.Column(db.String(200))
    workouts_completed = db.Column(db.Integer, default=0)
    avg_calories = db.Column(db.Float)
    adherence_score = db.Column(db.Float)
    
    # Agent predictions
    predicted_weight = db.Column(db.Float)
    
    notes_encrypted = db.Column(db.Text)
    
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def set_weight(self, weight):
        self.weight_encrypted = encrypt_data(str(weight))
    
    def get_weight(self):
        return float(decrypt_data(self.weight_encrypted)) if self.weight_encrypted else None
    
    def set_notes(self, notes):
        if notes:
            self.notes_encrypted = encrypt_data(notes)
    
    def get_notes(self):
        return decrypt_data(self.notes_encrypted) if self.notes_encrypted else None
    
    def to_dict(self):
        return {
            'id': self.id,
            'week_number': self.week_number,
            'date': self.date.isoformat(),
            'weight': self.get_weight(),
            'workouts_completed': self.workouts_completed,
            'avg_calories': self.avg_calories,
            'adherence_score': self.adherence_score,
            'predicted_weight': self.predicted_weight,
            'notes': self.get_notes()
        }
class ExerciseLog(db.Model):
    """Daily exercise logging"""
    __tablename__ = 'exercise_logs'

    id               = db.Column(db.Integer, primary_key=True)
    user_id          = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    date             = db.Column(db.Date, nullable=False)
    exercise_name    = db.Column(db.String(200), nullable=False)
    body_part        = db.Column(db.String(50))   # Chest, Back, Legs, etc.
    weight_kg        = db.Column(db.Float, default=0)
    sets_reps        = db.Column(db.String(50))   # e.g. "3x10"
    workout_duration = db.Column(db.Integer, default=0)  # minutes
    calories_burned  = db.Column(db.Float, default=0)
    timestamp        = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id':               self.id,
            'date':             self.date.isoformat(),
            'exercise_name':    self.exercise_name,
            'body_part':        self.body_part,
            'weight_kg':        self.weight_kg,
            'sets_reps':        self.sets_reps,
            'workout_duration': self.workout_duration,
            'calories_burned':  self.calories_burned,
            'timestamp':        self.timestamp.isoformat()
        }

# DATABASE INITIALIZATION
def init_db(app):
    """Initialize database with Flask app"""
    db.init_app(app)
    
    with app.app_context():
        # Create all tables
        db.create_all()
        print("Database tables created")


def reset_db(app):
    """Reset database (CAUTION: Deletes all data!)"""
    with app.app_context():
        db.drop_all()
        db.create_all()
        print("Database reset complete")