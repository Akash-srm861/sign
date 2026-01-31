"""
Simplified Flask application for Sign Language Learning App.
Integrated with Supabase PostgreSQL.
"""

import os
import sys

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def create_app():
    """Create and configure Flask application."""
    
    # Create Flask app
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'False') == 'True'
    
    # Initialize CORS (allow all origins for now - restrict in production)
    CORS(app, resources={r"/*": {"origins": "*"}})
    print("✓ CORS initialized")
    
    # Initialize database
    try:
        from database import init_db, test_connection
        if test_connection():
            print("✓ Database connected")
        else:
            print("⚠ Database connection failed - running without persistence")
    except Exception as e:
        print(f"⚠ Database initialization error: {e}")
    
    # Register blueprints
    from routes.api_routes import api_bp
    from routes.chatbot_routes import chatbot_bp
    from routes.auth_routes import auth_bp
    
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(chatbot_bp, url_prefix='/api/chatbot')
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    print("✓ Routes registered")
    
    # Health check
    @app.route('/health')
    def health():
        return jsonify({
            'status': 'ok',
            'message': 'Server is running',
            'models_loaded': True
        })
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'success': False,
            'error': 'Resource not found'
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500
    
    return app


if __name__ == '__main__':
    app = create_app()
    port = int(os.getenv('PORT', 5000))
    print(f"\n{'='*60}")
    print(f"Sign Language Learning App - Backend Server")
    print(f"{'='*60}")
    print(f"Server running on: http://localhost:{port}")
    print(f"Health check: http://localhost:{port}/health")
    print(f"API endpoints: http://localhost:{port}/api/...")
    print(f"{'='*60}\n")
    
    app.run(host='0.0.0.0', port=port, debug=False)
