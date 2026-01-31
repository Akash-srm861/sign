"""
Main API routes for sign detection and progress tracking.
Handles sign classification, progress updates, and learning modules.
"""

from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import database models and config
from database import SessionLocal, User, Progress, Session as DBSession
from config import Config

# Create blueprint
api_bp = Blueprint('api', __name__)

# Initialize ML components (lazy loading)
hand_detector = None
sign_classifier = None


def get_hand_detector():
    """Get or create hand detector instance."""
    global hand_detector
    if hand_detector is None:
        from ml.hand_detector import HandDetector
        hand_detector = HandDetector(
            max_hands=2,
            detection_confidence=0.3,  # Lowered from 0.7 for better sensitivity
            tracking_confidence=0.3    # Lowered from 0.5 for better sensitivity
        )
    return hand_detector


def get_sign_classifier():
    """Get or create sign classifier instance."""
    global sign_classifier
    if sign_classifier is None:
        from ml.classifier import SignClassifier
        sign_classifier = SignClassifier()
    return sign_classifier


# ==================== Sign Detection Routes ====================

@api_bp.route('/detect', methods=['POST'])
def detect_sign():
    """
    Detect hand landmarks and classify sign from image.
    
    Expected JSON body:
        - image: Base64-encoded image data
        - target_sign (optional): Expected sign for validation
    
    Returns:
        JSON with detection results and classification
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image data provided'
            }), 400
        
        image_data = data['image']
        target_sign = data.get('target_sign')
        
        # Detect hands
        detector = get_hand_detector()
        detection_result = detector.detect_from_base64(image_data)
        
        print(f"🔍 Detection result: hands_detected={detection_result.get('hands_detected', 0)}")
        
        if not detection_result['success']:
            return jsonify(detection_result), 400
        
        if detection_result['hands_detected'] == 0:
            return jsonify({
                'success': True,
                'hands_detected': False,
                'message': 'No hands detected'
            }), 200
        
        # Classify the detected sign
        classifier = get_sign_classifier()
        landmarks = detection_result['landmarks'][0]
        
        if target_sign:
            classification = classifier.validate_sign(
                landmarks,
                target_sign
            )
            print(f"🎯 Classification for target '{target_sign}': {classification}")
        else:
            classification = classifier.classify(landmarks)
            print(f"🎯 Classification: {classification}")
        
        return jsonify({
            'success': True,
            'hands_detected': True,
            'num_hands': detection_result['hands_detected'],
            'landmarks': detection_result['landmarks'],
            'classification': classification
        }), 200
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Detection error: {error_trace}")
        return jsonify({
            'success': False,
            'error': f'Detection failed: {str(e)}'
        }), 500


@api_bp.route('/detect/landmarks', methods=['POST'])
def detect_landmarks_only():
    """
    Detect hand landmarks without classification.
    Useful for debugging or visualization.
    
    Expected JSON body:
        - image: Base64-encoded image data
    
    Returns:
        JSON with raw landmark data
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image data provided'
            }), 400
        
        detector = get_hand_detector()
        result = detector.detect_from_base64(data['image'])
        
        return jsonify(result), 200 if result['success'] else 400
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Detection failed: {str(e)}'
        }), 500


# ==================== Progress Routes ====================

@api_bp.route('/progress', methods=['GET'])
def get_progress():
    """
    Get user's overall progress across all categories.
    
    Returns:
        JSON with progress statistics and per-sign data
    """
    try:
        # Get user_id from query parameter or request body
        user_id = request.args.get('user_id') or request.get_json().get('user_id')
        if not user_id:
            return jsonify({'success': False, 'error': 'user_id is required'}), 400
        
        # Get all progress records for user
        db = SessionLocal()
        progress_records = db.query(Progress).filter_by(user_id=user_id).all()
        
        # Calculate overall statistics
        total_attempts = sum(p.attempts for p in progress_records)
        total_correct = sum(p.successes for p in progress_records)
        overall_accuracy = (total_correct / total_attempts * 100) if total_attempts > 0 else 0
        
        # Simple progress overview - no category breakdown
        return jsonify({
            'success': True,
            'overview': {
                'total_signs_practiced': len(progress_records),
                'total_attempts': total_attempts,
                'total_correct': total_correct,
                'overall_accuracy': round(overall_accuracy, 1)
            },
            'signs': [{
                'sign': p.sign,
                'accuracy': p.accuracy,
                'attempts': p.attempts,
                'is_mastered': p.is_mastered
            } for p in progress_records]
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get progress: {str(e)}'
        }), 500
    finally:
        if 'db' in locals():
            db.close()


@api_bp.route('/progress/<category>', methods=['GET'])

def get_category_progress(category):
    """
    Get user's progress for a specific category.
    
    Args:
        category: Category name (alphabets, numbers, words)
    
    Returns:
        JSON with category-specific progress data
    """
    try:
        # Get user_id from query parameter
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'error': 'user_id is required'}), 400
        
        if category not in Config.SIGN_CATEGORIES:
            return jsonify({
                'success': False,
                'error': f'Invalid category: {category}'
            }), 400
        
        # Get all signs in category
        all_signs = Config.SIGN_CATEGORIES[category]
        
        # Get progress records
        db = SessionLocal()
        progress_records = db.query(Progress).filter_by(
            user_id=user_id
        ).filter(Progress.sign.in_(all_signs)).all()
        
        progress_map = {p.sign_name: p for p in progress_records}
        
        # Build complete sign list with progress
        signs_data = []
        for sign in all_signs:
            if sign in progress_map:
                signs_data.append(progress_map[sign].to_dict())
            else:
                signs_data.append({
                    'sign_name': sign,
                    'category': category,
                    'attempts': 0,
                    'correct_attempts': 0,
                    'accuracy': 0,
                    'mastery_level': 0,
                    'last_practiced': None
                })
        
        return jsonify({
            'success': True,
            'category': category,
            'total_signs': len(all_signs),
            'signs_practiced': len(progress_records),
            'signs': signs_data
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get category progress: {str(e)}'
        }), 500
    finally:
        if 'db' in locals():
            db.close()


@api_bp.route('/progress/record', methods=['POST'])

def record_attempt():
    """
    Record a practice attempt for a sign.
    
    Expected JSON body:
        - category: Sign category
        - sign_name: Name of the sign
        - correct: Boolean indicating if attempt was correct
        - session_id (optional): Current learning session ID
    
    Returns:
        JSON with updated progress data
    """
    try:
        data = request.get_json()
        
        user_id = data.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'error': 'user_id is required'}), 400
        
        category = data.get('category')
        sign_name = data.get('sign_name')
        correct = data.get('correct', False)
        session_id = data.get('session_id')
        
        if not category or not sign_name:
            return jsonify({
                'success': False,
                'error': 'Category and sign_name are required'
            }), 400
        
        # Get or create progress record
        db = SessionLocal()
        progress = db.query(Progress).filter_by(
            user_id=user_id,
            sign=sign_name
        ).first()
        
        if not progress:
            progress = Progress(
                user_id=user_id,
                sign=sign_name
            )
            db.add(progress)
        
        # Update progress
        progress.attempts += 1
        if correct:
            progress.correct_attempts += 1
        
        progress.last_practiced = datetime.utcnow()
        # Use property for accuracy calculation
        progress.is_mastered = progress.accuracy >= 80 and progress.attempts >= 10
        
        db.commit()
        
        return jsonify({
            'success': True,
            'progress': {
                'id': progress.id,
                'sign': progress.sign_name,
                'accuracy': progress.accuracy,
                'attempts': progress.attempts,
                'successes': progress.correct_attempts,
                'is_mastered': progress.is_mastered
            }
        }), 200
        
    except Exception as e:
        if 'db' in locals():
            db.rollback()
        return jsonify({
            'success': False,
            'error': f'Failed to record progress: {str(e)}'
        }), 500
    finally:
        if 'db' in locals():
            db.close()


# ==================== Session Routes ====================

@api_bp.route('/session/start', methods=['POST'])
def start_session():
    """
    Start a new learning session.
    
    Expected JSON body:
        - category: Category to practice
    
    Returns:
        JSON with new session data
    """
    try:
        data = request.get_json()
        
        user_id = data.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'error': 'user_id is required'}), 400
        
        category = data.get('category')
        if not category:
            return jsonify({
                'success': False,
                'error': 'Category is required'
            }), 400
        
        # Create new session
        import uuid
        db = SessionLocal()
        session = DBSession(
            user_id=user_id,
            session_id=str(uuid.uuid4()),
            start_time=datetime.utcnow()
        )
        db.add(session)
        db.commit()
        
        return jsonify({
            'success': True,
            'session': {
                'id': session.id,
                'session_id': session.session_id,
                'user_id': session.user_id,
                'start_time': session.start_time.isoformat()
            }
        }), 201
        
    except Exception as e:
        if 'db' in locals():
            db.rollback()
        return jsonify({
            'success': False,
            'error': f'Failed to start session: {str(e)}'
        }), 500
    finally:
        if 'db' in locals():
            db.close()


@api_bp.route('/session/<int:session_id>/end', methods=['POST'])

def end_session(session_id):
    """
    End a learning session.
    
    Args:
        session_id: ID of the session to end
    
    Returns:
        JSON with final session data
    """
    try:
        data = request.get_json() or {}
        user_id = data.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'error': 'user_id is required'}), 400
        
        db = SessionLocal()
        session = db.query(DBSession).get(session_id)
        
        if not session:
            return jsonify({
                'success': False,
                'error': 'Session not found'
            }), 404
        
        if session.user_id != user_id:
            return jsonify({
                'success': False,
                'error': 'Unauthorized'
            }), 403
        
        session.end_time = datetime.utcnow()
        db.commit()
        
        return jsonify({
            'success': True,
            'session': {
                'id': session.id,
                'session_id': session.session_id,
                'end_time': session.end_time.isoformat() if session.end_time else None
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to end session: {str(e)}'
        }), 500
    finally:
        if 'db' in locals():
            db.close()


# ==================== Learning Module Routes ====================

@api_bp.route('/signs/<category>', methods=['GET'])
def get_signs(category):
    """
    Get all signs for a category.
    
    Args:
        category: Category name (alphabets, numbers, words)
    
    Returns:
        JSON with list of signs and their descriptions
    """
    try:
        import traceback
        print(f"📋 Loading signs for category: {category}")
        
        if category not in Config.SIGN_CATEGORIES:
            return jsonify({
                'success': False,
                'error': f'Invalid category: {category}'
            }), 400
        
        signs = Config.SIGN_CATEGORIES[category]
        print(f"✓ Found {len(signs)} signs in category")
        
        # Get hints for each sign
        print("Loading classifier...")
        classifier = get_sign_classifier()
        print("✓ Classifier loaded")
        
        signs_data = []
        for sign in signs:
            sign_upper = sign.upper()
            sign_lower = sign.lower()
            
            # Multiple image sources with fallbacks (most reliable first)
            image_urls = [
                # Local images first (if available)
                f"/images/signs/{sign_lower}.png",
                # Public CDN sources
                f"https://www.lifeprint.com/asl101/images-signs/{sign_lower}.jpg",
                f"https://www.signingsavvy.com/images/words/alphabet/{sign_lower}/1.jpg",
            ]
            
            signs_data.append({
                'name': sign,
                'category': category,
                'hint': classifier._get_hint(sign),
                'imageUrl': image_urls[0],  # Primary source
                'imageFallbacks': image_urls[1:],  # Backup sources
            })
        
        print(f"✓ Prepared {len(signs_data)} signs")
        
        return jsonify({
            'success': True,
            'category': category,
            'signs': signs_data
        }), 200
        
    except Exception as e:
        import traceback
        print(f"❌ Error in get_signs: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Failed to get signs: {str(e)}'
        }), 500


@api_bp.route('/dashboard/stats', methods=['GET'])
def get_dashboard_stats():
    """
    Get dashboard statistics for the current user.
    
    Returns:
        JSON with comprehensive statistics
    """
    try:
        # Get user_id from query parameter
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'error': 'user_id is required'}), 400
        
        # Get user
        db = SessionLocal()
        user = db.query(User).get(user_id)
        
        # Calculate streak (days practiced in a row)
        sessions = db.query(DBSession).filter_by(user_id=user_id)\
            .order_by(DBSession.start_time.desc()).all()
        
        streak = 0
        if sessions:
            current_date = datetime.utcnow().date()
            for session in sessions:
                session_date = session.start_time.date()
                if session_date == current_date - timedelta(days=streak):
                    streak += 1
                elif session_date < current_date - timedelta(days=streak):
                    break
        
        # Get weekly activity
        week_ago = datetime.utcnow() - timedelta(days=7)
        weekly_sessions = db.query(DBSession).filter(
            DBSession.user_id == user_id,
            DBSession.start_time >= week_ago
        ).all()
        
        weekly_practice_time = len(weekly_sessions)  # Simple count for now
        weekly_signs_practiced = sum(s.total_signs_practiced or 0 for s in weekly_sessions)
        
        # Get mastery levels
        progress_records = db.query(Progress).filter_by(user_id=user_id).all()
        
        mastered = len([p for p in progress_records if p.is_mastered])
        learning = len([p for p in progress_records if 50 <= p.accuracy < 80])
        beginner = len([p for p in progress_records if p.accuracy < 50 and p.attempts > 0])
        
        return jsonify({
            'success': True,
            'stats': {
                'streak': streak,
                'total_signs_learned': len(progress_records),
                'mastered_signs': mastered,
                'learning_signs': learning,
                'beginner_signs': beginner,
                'weekly_practice_time': round(weekly_practice_time, 1),
                'weekly_signs_practiced': weekly_signs_practiced,
                'total_sessions': len(sessions),
                'member_since': user.created_at.isoformat() if user else None
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get stats: {str(e)}'
        }), 500
    finally:
        if 'db' in locals():
            db.close()
