"""
Progress and learning session models.
Tracks user learning progress and session history.
"""

from datetime import datetime
from . import db


class Progress(db.Model):
    """
    Progress model tracking user's learning progress for each sign.
    
    Attributes:
        id: Unique identifier
        user_id: Reference to the user
        category: Sign category (alphabets/numbers/words)
        sign_name: Name of the sign
        attempts: Total number of attempts
        correct_attempts: Number of correct attempts
        mastery_level: Current mastery level (0-100)
        last_practiced: Last practice timestamp
    """
    
    __tablename__ = 'progress'
    
    # Primary key
    id = db.Column(db.Integer, primary_key=True)
    
    # Foreign key
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    
    # Progress data
    category = db.Column(db.String(50), nullable=False)  # alphabets, numbers, words
    sign_name = db.Column(db.String(100), nullable=False)
    attempts = db.Column(db.Integer, default=0)
    correct_attempts = db.Column(db.Integer, default=0)
    mastery_level = db.Column(db.Float, default=0.0)  # 0-100 percentage
    
    # Timestamps
    last_practiced = db.Column(db.DateTime, default=datetime.utcnow)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Unique constraint for user + category + sign combination
    __table_args__ = (
        db.UniqueConstraint('user_id', 'category', 'sign_name', name='unique_user_sign'),
    )
    
    def __init__(self, user_id, category, sign_name):
        """
        Initialize progress tracking for a specific sign.
        
        Args:
            user_id: ID of the user
            category: Category of the sign
            sign_name: Name of the sign
        """
        self.user_id = user_id
        self.category = category
        self.sign_name = sign_name
    
    def record_attempt(self, correct):
        """
        Record a practice attempt and update mastery level.
        
        Args:
            correct: Boolean indicating if the attempt was correct
        """
        self.attempts += 1
        if correct:
            self.correct_attempts += 1
        
        # Calculate mastery level using weighted recent performance
        # Recent attempts matter more than older ones
        if self.attempts > 0:
            base_accuracy = (self.correct_attempts / self.attempts) * 100
            # Bonus for consistency (reaching certain thresholds)
            consistency_bonus = min(self.attempts / 10, 10)  # Max 10 bonus points
            self.mastery_level = min(base_accuracy + consistency_bonus, 100)
        
        self.last_practiced = datetime.utcnow()
        db.session.commit()
    
    @property
    def accuracy(self):
        """Calculate accuracy percentage."""
        if self.attempts == 0:
            return 0
        return round((self.correct_attempts / self.attempts) * 100, 1)
    
    def to_dict(self):
        """
        Convert progress to dictionary for JSON serialization.
        
        Returns:
            dict: Progress data
        """
        return {
            'id': self.id,
            'category': self.category,
            'sign_name': self.sign_name,
            'attempts': self.attempts,
            'correct_attempts': self.correct_attempts,
            'accuracy': self.accuracy,
            'mastery_level': round(self.mastery_level, 1),
            'last_practiced': self.last_practiced.isoformat() if self.last_practiced else None
        }
    
    def __repr__(self):
        """String representation of the progress."""
        return f'<Progress {self.sign_name}: {self.mastery_level}%>'


class LearningSession(db.Model):
    """
    Learning session model tracking practice sessions.
    
    Attributes:
        id: Unique identifier
        user_id: Reference to the user
        category: Category practiced
        started_at: Session start time
        ended_at: Session end time
        total_signs: Number of signs practiced
        correct_signs: Number of correct signs
        duration_minutes: Session duration in minutes
    """
    
    __tablename__ = 'learning_sessions'
    
    # Primary key
    id = db.Column(db.Integer, primary_key=True)
    
    # Foreign key
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    
    # Session data
    category = db.Column(db.String(50), nullable=False)
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    ended_at = db.Column(db.DateTime, nullable=True)
    total_signs = db.Column(db.Integer, default=0)
    correct_signs = db.Column(db.Integer, default=0)
    duration_minutes = db.Column(db.Float, default=0)
    
    def __init__(self, user_id, category):
        """
        Start a new learning session.
        
        Args:
            user_id: ID of the user
            category: Category being practiced
        """
        self.user_id = user_id
        self.category = category
        self.started_at = datetime.utcnow()
    
    def end_session(self):
        """End the current session and calculate duration."""
        self.ended_at = datetime.utcnow()
        delta = self.ended_at - self.started_at
        self.duration_minutes = delta.total_seconds() / 60
        db.session.commit()
    
    def record_sign(self, correct):
        """
        Record a sign attempt in this session.
        
        Args:
            correct: Boolean indicating if the sign was correct
        """
        self.total_signs += 1
        if correct:
            self.correct_signs += 1
        db.session.commit()
    
    @property
    def accuracy(self):
        """Calculate session accuracy percentage."""
        if self.total_signs == 0:
            return 0
        return round((self.correct_signs / self.total_signs) * 100, 1)
    
    def to_dict(self):
        """
        Convert session to dictionary for JSON serialization.
        
        Returns:
            dict: Session data
        """
        return {
            'id': self.id,
            'category': self.category,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'ended_at': self.ended_at.isoformat() if self.ended_at else None,
            'total_signs': self.total_signs,
            'correct_signs': self.correct_signs,
            'accuracy': self.accuracy,
            'duration_minutes': round(self.duration_minutes, 1) if self.duration_minutes else 0
        }
    
    def __repr__(self):
        """String representation of the session."""
        return f'<LearningSession {self.category}: {self.total_signs} signs>'
