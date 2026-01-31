"""
Database configuration and models for Supabase PostgreSQL.
"""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

# Database connection
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL, pool_pre_ping=True, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    """User model for sign language learners."""
    __tablename__ = 'users'
    
    id = Column(String(255), primary_key=True)  # Supabase Auth user ID
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password = Column(String(255), nullable=False)  # Store password hash
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    total_signs_learned = Column(Integer, default=0)
    current_streak = Column(Integer, default=0)
    last_practice_date = Column(DateTime)


class Progress(Base):
    """User progress tracking for each sign."""
    __tablename__ = 'progress'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False, index=True)
    sign = Column(String(50), nullable=False)
    accuracy = Column(Float, default=0.0)
    attempts = Column(Integer, default=0)
    successes = Column(Integer, default=0)
    last_practiced = Column(DateTime, default=datetime.utcnow)
    is_mastered = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Session(Base):
    """Practice session tracking."""
    __tablename__ = 'sessions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False, index=True)
    session_id = Column(String(100), unique=True, nullable=False)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    total_signs_practiced = Column(Integer, default=0)
    correct_signs = Column(Integer, default=0)
    average_confidence = Column(Float, default=0.0)
    session_data = Column(JSON)  # Store detailed session metrics


class Achievement(Base):
    """User achievements and badges."""
    __tablename__ = 'achievements'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False, index=True)
    achievement_type = Column(String(50), nullable=False)  # e.g., 'first_sign', 'streak_7', 'master_10'
    achievement_name = Column(String(100), nullable=False)
    earned_at = Column(DateTime, default=datetime.utcnow)
    achievement_data = Column(JSON)


class ChatHistory(Base):
    """Chatbot conversation history."""
    __tablename__ = 'chat_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False, index=True)
    message = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    context = Column(JSON)  # Store conversation context


def init_db():
    """Initialize database tables."""
    try:
        Base.metadata.create_all(bind=engine)
        print("✓ Database tables created successfully")
        return True
    except Exception as e:
        print(f"✗ Error creating database tables: {e}")
        return False


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_connection():
    """Test database connection."""
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✓ Database connection successful")
            return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing Supabase PostgreSQL connection...")
    test_connection()
    print("\nInitializing database tables...")
    init_db()
