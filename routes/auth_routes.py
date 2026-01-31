"""
Authentication routes for user signup and login.
Stores users in Supabase PostgreSQL database.
"""

from flask import Blueprint, request, jsonify
from datetime import datetime
from database import SessionLocal, User
import hashlib

auth_bp = Blueprint('auth', __name__)

def hash_password(password):
    """Simple password hashing"""
    return hashlib.sha256(password.encode()).hexdigest()

@auth_bp.route('/signup', methods=['POST'])
def signup():
    """Create a new user account"""
    try:
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        if not username or not email or not password:
            return jsonify({
                'success': False,
                'error': 'Username, email and password are required'
            }), 400
        
        db = SessionLocal()
        
        try:
            # Check if user already exists
            existing_user = db.query(User).filter(
                (User.email == email) | (User.username == username)
            ).first()
            
            if existing_user:
                return jsonify({
                    'success': False,
                    'error': 'Email or username already exists'
                }), 409
            
            # Create new user with password stored in database
            user_id = f"user_{int(datetime.utcnow().timestamp() * 1000)}"
            new_user = User(
                id=user_id,
                username=username,
                email=email,
                password=hash_password(password),  # Store hashed password
                created_at=datetime.utcnow()
            )
            
            db.add(new_user)
            db.commit()
            
            user_data = {
                'id': new_user.id,
                'username': new_user.username,
                'email': new_user.email
            }
            
            print(f"✓ User created in Supabase: {email}")
            
            return jsonify({
                'success': True,
                'user': user_data
            }), 201
            
        finally:
            db.close()
        
    except Exception as e:
        print(f"Signup error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@auth_bp.route('/login', methods=['POST'])
def login():
    """Log in an existing user"""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({
                'success': False,
                'error': 'Email and password are required'
            }), 400
        
        db = SessionLocal()
        
        try:
            # Get user from database
            user = db.query(User).filter(User.email == email).first()
            
            if not user:
                return jsonify({
                    'success': False,
                    'error': 'Invalid email or password'
                }), 401
            
            # Check password
            if user.password != hash_password(password):
                return jsonify({
                    'success': False,
                    'error': 'Invalid email or password'
                }), 401
            
            user_data = {
                'id': user.id,
                'username': user.username,
                'email': user.email
            }
            
            print(f"✓ User logged in: {email}")
            
            return jsonify({
                'success': True,
                'user': user_data
            }), 200
            
        finally:
            db.close()
        
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
