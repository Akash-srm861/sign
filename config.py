"""
Configuration settings for the Sign Language Learning Application.
Contains all configurable parameters for the application.
"""

import os
from datetime import timedelta

class Config:
    """Base configuration class with default settings."""
    
    # Flask Settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
    DEBUG = False
    TESTING = False
    
    # Database Settings
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///sign_language.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # JWT Settings
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'jwt-secret-key-change-in-production')
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)
    
    # CORS Settings
    CORS_ORIGINS = ['http://localhost:8080', 'http://127.0.0.1:8080', 'http://localhost:5500']
    
    # MediaPipe Settings
    MEDIAPIPE_MAX_HANDS = 2
    MEDIAPIPE_DETECTION_CONFIDENCE = 0.7
    MEDIAPIPE_TRACKING_CONFIDENCE = 0.5
    
    # ML Model Settings
    MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'trained_models')
    SVM_MODEL_FILE = 'svm_motion_model.joblib'
    RF_MODEL_FILE = 'rf_shape_model.joblib'
    
    # Sign Categories
    SIGN_CATEGORIES = {
        'alphabets': list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
        'numbers': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        'words': ['hello', 'thank you', 'please', 'sorry', 'yes', 'no', 'help', 'love', 'friend', 'family']
    }
    
    # Preprocessing Settings
    LANDMARK_NORMALIZATION = True
    NOISE_THRESHOLD = 0.02
    SMOOTHING_WINDOW = 5
    
    # OpenAI Settings (for chatbot)
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')


class DevelopmentConfig(Config):
    """Development configuration with debug enabled."""
    DEBUG = False
    SQLALCHEMY_ECHO = True


class ProductionConfig(Config):
    """Production configuration with security settings."""
    DEBUG = False
    # Add production-specific settings here


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'


# Configuration dictionary for easy access
config_dict = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config():
    """Get the current configuration based on environment."""
    env = os.environ.get('FLASK_ENV', 'development')
    return config_dict.get(env, DevelopmentConfig)
