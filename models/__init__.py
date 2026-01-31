"""
Database models package.
Contains SQLAlchemy models for user data and progress tracking.
"""

from flask_sqlalchemy import SQLAlchemy

# Initialize SQLAlchemy instance
db = SQLAlchemy()

# Import models for easy access
from .user import User
from .progress import Progress, LearningSession

__all__ = ['db', 'User', 'Progress', 'LearningSession']
