"""
User model for authentication and profile management.
Handles user registration, login, and profile data.
"""

from datetime import datetime
from . import db
import bcrypt


class User(db.Model):
    """
    User model representing registered users of the application.
    
    Attributes:
        id: Unique identifier for the user
        username: User's display name
        email: User's email address (unique)
        password_hash: Hashed password for security
        created_at: Account creation timestamp
        last_login: Last login timestamp
        is_active: Whether the account is active
        avatar_url: URL to user's avatar image
        preferred_hand: User's dominant hand (left/right)
    """
    
    __tablename__ = 'users'
    
    # Primary key
    id = db.Column(db.Integer, primary_key=True)
    
    # User credentials
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Profile settings
    is_active = db.Column(db.Boolean, default=True)
    avatar_url = db.Column(db.String(255), default='')
    preferred_hand = db.Column(db.String(10), default='right')
    
    # Relationships
    progress = db.relationship('Progress', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    sessions = db.relationship('LearningSession', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
    def __init__(self, username, email, password):
        """
        Initialize a new user with hashed password.
        
        Args:
            username: User's display name
            email: User's email address
            password: Plain text password (will be hashed)
        """
        self.username = username
        self.email = email.lower()
        self.set_password(password)
    
    def set_password(self, password):
        """
        Hash and set the user's password.
        
        Args:
            password: Plain text password to hash
        """
        salt = bcrypt.gensalt()
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def check_password(self, password):
        """
        Verify a password against the stored hash.
        
        Args:
            password: Plain text password to verify
            
        Returns:
            bool: True if password matches, False otherwise
        """
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))
    
    def update_last_login(self):
        """Update the last login timestamp to now."""
        self.last_login = datetime.utcnow()
        db.session.commit()
    
    def to_dict(self):
        """
        Convert user to dictionary for JSON serialization.
        
        Returns:
            dict: User data (excluding sensitive information)
        """
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'avatar_url': self.avatar_url,
            'preferred_hand': self.preferred_hand
        }
    
    def __repr__(self):
        """String representation of the user."""
        return f'<User {self.username}>'
