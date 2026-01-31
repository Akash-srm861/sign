"""
Chatbot routes for AI learning assistant.
Provides conversational help for learning sign language.
"""

from flask import Blueprint, request, jsonify
from database import SessionLocal, ChatHistory
from datetime import datetime
import re

# Create blueprint
chatbot_bp = Blueprint('chatbot', __name__)


class SignLanguageChatbot:
    """
    AI chatbot for sign language learning assistance.
    
    Uses rule-based responses with optional OpenAI integration
    for more complex queries.
    """
    
    def __init__(self):
        """Initialize the chatbot with knowledge base."""
        self.knowledge_base = self._build_knowledge_base()
        self.conversation_history = {}  # Store per-user conversations
    
    def _build_knowledge_base(self):
        """Build the knowledge base for sign language queries."""
        return {
            # Greetings
            'greetings': {
                'patterns': ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening'],
                'responses': [
                    "Hello! 👋 I'm your sign language learning assistant. How can I help you today?",
                    "Hi there! Ready to learn some sign language? Ask me anything!",
                    "Hey! Great to see you. What would you like to learn about sign language?"
                ]
            },
            
            # ASL Alphabet
            'alphabet': {
                'patterns': ['alphabet', 'letters', 'abc', 'finger spelling', 'fingerspelling'],
                'responses': [
                    "The ASL alphabet uses 26 hand shapes to represent each letter. Here are some tips:\n\n"
                    "• Keep your hand at shoulder height\n"
                    "• Face your palm outward (except for some letters)\n"
                    "• Practice each letter slowly before speeding up\n\n"
                    "Would you like tips for a specific letter?"
                ]
            },
            
            # Numbers
            'numbers': {
                'patterns': ['numbers', 'counting', 'digits', 'how to sign number'],
                'responses': [
                    "ASL numbers 1-5 are shown on one hand with fingers extended:\n\n"
                    "• 1: Index finger up\n"
                    "• 2: Index and middle fingers up (like peace sign)\n"
                    "• 3: Thumb, index, and middle fingers up\n"
                    "• 4: Four fingers up, thumb tucked\n"
                    "• 5: All five fingers spread\n\n"
                    "Numbers 6-10 use different hand positions. Want to learn more?"
                ]
            },
            
            # Common phrases
            'phrases': {
                'patterns': ['common phrase', 'basic phrase', 'everyday sign', 'useful sign'],
                'responses': [
                    "Here are some essential ASL phrases to learn:\n\n"
                    "• **Hello**: Wave your open hand\n"
                    "• **Thank you**: Touch chin, move hand forward\n"
                    "• **Please**: Rub chest in circular motion\n"
                    "• **Sorry**: Make fist, rub on chest\n"
                    "• **Yes**: Nod your fist\n"
                    "• **No**: Snap index and middle finger to thumb\n\n"
                    "Would you like detailed instructions for any of these?"
                ]
            },
            
            # Learning tips
            'tips': {
                'patterns': ['tip', 'advice', 'how to learn', 'practice', 'improve'],
                'responses': [
                    "Here are my top tips for learning sign language:\n\n"
                    "1. **Practice daily** - Even 10-15 minutes helps!\n"
                    "2. **Use a mirror** - Watch your own signing\n"
                    "3. **Learn in context** - Practice whole phrases, not just words\n"
                    "4. **Watch native signers** - Videos help with natural movement\n"
                    "5. **Be patient** - Learning any language takes time\n\n"
                    "What aspect would you like to focus on?"
                ]
            },
            
            # Specific letters
            'letter_a': {
                'patterns': ['sign a', 'letter a', 'how to sign a'],
                'responses': ["To sign the letter **A**: Make a fist with your thumb resting on the side of your hand, facing outward. Keep your fingers tightly closed."]
            },
            'letter_b': {
                'patterns': ['sign b', 'letter b', 'how to sign b'],
                'responses': ["To sign the letter **B**: Hold your hand flat with all fingers together pointing up, and tuck your thumb across your palm."]
            },
            'letter_c': {
                'patterns': ['sign c', 'letter c', 'how to sign c'],
                'responses': ["To sign the letter **C**: Curve your hand like you're holding a cup or the letter C shape. Thumb and fingers don't touch."]
            },
            
            # Technical questions
            'how_it_works': {
                'patterns': ['how does this work', 'how does detection work', 'technology', 'ai', 'machine learning'],
                'responses': [
                    "Great question! This app uses:\n\n"
                    "• **MediaPipe**: Detects 21 hand landmarks in real-time\n"
                    "• **Preprocessing**: Normalizes and cleans the landmark data\n"
                    "• **Machine Learning**: SVM for motion signs, Random Forest for static signs\n"
                    "• **Real-time feedback**: Compares your sign to the expected pattern\n\n"
                    "The more you practice, the better you'll get! 🎯"
                ]
            },
            
            # Help
            'help': {
                'patterns': ['help', 'what can you do', 'features', 'commands'],
                'responses': [
                    "I can help you with:\n\n"
                    "📚 **Learning**: Ask about letters, numbers, or phrases\n"
                    "💡 **Tips**: Get advice on how to improve\n"
                    "❓ **Questions**: Answer your sign language questions\n"
                    "🎯 **Practice**: Guide you through exercises\n"
                    "📊 **Progress**: Discuss your learning journey\n\n"
                    "Just type your question and I'll do my best to help!"
                ]
            },
            
            # Encouragement
            'encouragement': {
                'patterns': ['difficult', 'hard', 'struggle', 'can\'t', 'frustrated', 'give up'],
                'responses': [
                    "Learning sign language can be challenging, but don't give up! 💪\n\n"
                    "Remember:\n"
                    "• Everyone learns at their own pace\n"
                    "• Mistakes are part of learning\n"
                    "• Even small progress is progress\n"
                    "• Practice makes perfect\n\n"
                    "What specific sign are you finding difficult? I can help!"
                ]
            },
            
            # Default
            'default': {
                'patterns': [],
                'responses': [
                    "I'm not sure I understand that question. Could you rephrase it?\n\n"
                    "You can ask me about:\n"
                    "• Specific letters or numbers\n"
                    "• Common phrases\n"
                    "• Learning tips\n"
                    "• How the app works",
                    
                    "Hmm, I'm not sure about that. Try asking about:\n"
                    "• How to sign specific letters\n"
                    "• Tips for learning\n"
                    "• Common ASL phrases"
                ]
            }
        }
    
    def get_response(self, message: str, user_id: str = None) -> dict:
        """
        Generate a response to the user's message.
        
        Args:
            message: User's input message
            user_id: Optional user ID for conversation context
            
        Returns:
            dict: Response with text and metadata
        """
        message_lower = message.lower().strip()
        
        # Find matching category
        best_match = None
        for category, data in self.knowledge_base.items():
            if category == 'default':
                continue
            for pattern in data['patterns']:
                if pattern in message_lower:
                    best_match = category
                    break
            if best_match:
                break
        
        # Get response
        response_image = None
        detected_letter = None
        
        if best_match:
            import random
            responses = self.knowledge_base[best_match]['responses']
            response_text = random.choice(responses)
            
            # Check if this is a specific letter category
            if best_match.startswith('letter_'):
                detected_letter = best_match.split('_')[1].upper()
        else:
            # Check for specific letter queries
            letter_match = re.search(r'(?:sign|letter)\s+([a-z])\b', message_lower)
            if letter_match:
                detected_letter = letter_match.group(1).upper()
                from ml import SignClassifier
                classifier = SignClassifier()
                hint = classifier._get_hint(detected_letter)
                response_text = f"To sign the letter **{detected_letter}**: {hint}"
            else:
                import random
                responses = self.knowledge_base['default']['responses']
                response_text = random.choice(responses)
        
        # Add image URL if a letter was detected
        if detected_letter:
            response_image = f"/images/signs/{detected_letter.lower()}.png"
        
        # Save conversation to database (Supabase)
        if user_id:
            try:
                db = SessionLocal()
                chat_entry = ChatHistory(
                    user_id=user_id,
                    message=message,
                    response=response_text,
                    timestamp=datetime.utcnow(),
                    context={
                        'category': best_match or 'default',
                        'letter': detected_letter,
                        'image': response_image
                    }
                )
                db.add(chat_entry)
                db.commit()
                db.close()
            except Exception as e:
                print(f"Failed to save chat history: {e}")
        
        result = {
            'success': True,
            'response': response_text,
            'category': best_match or 'default'
        }
        
        # Include image if available
        if response_image:
            result['image'] = response_image
            result['letter'] = detected_letter
        
        return result


@chatbot_bp.route('/chat', methods=['POST'])
def chat():
    """
    Chat endpoint.
    
    Body:
        - message: User message
        - user_id: User ID (optional, for history storage)
    
    Returns:
        JSON with chatbot response (history saved to Supabase)
    """
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'success': False,
                'error': 'No message provided'
            }), 400
        
        message = data['message']
        user_id = data.get('user_id')  # Get user_id from request
        
        # Get chatbot response (will save to Supabase if user_id provided)
        response = chatbot.get_response(message, user_id)
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Chat failed: {str(e)}'
        }), 500


@chatbot_bp.route('/history', methods=['GET'])
def get_chat_history():
    """
    Get chat history for a user from Supabase.
    
    Query params:
        - user_id: User ID
        - limit: Number of messages to retrieve (default: 50)
    
    Returns:
        JSON with chat history
    """
    try:
        user_id = request.args.get('user_id')
        limit = int(request.args.get('limit', 50))
        
        if not user_id:
            return jsonify({
                'success': False,
                'error': 'user_id is required'
            }), 400
        
        db = SessionLocal()
        history = db.query(ChatHistory).filter_by(
            user_id=user_id
        ).order_by(
            ChatHistory.timestamp.desc()
        ).limit(limit).all()
        
        chat_list = [{
            'id': chat.id,
            'message': chat.message,
            'response': chat.response,
            'timestamp': chat.timestamp.isoformat(),
            'context': chat.context
        } for chat in history]
        
        db.close()
        
        return jsonify({
            'success': True,
            'history': list(reversed(chat_list)),  # Oldest first
            'count': len(chat_list)
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get history: {str(e)}'
        }), 500


@chatbot_bp.route('/suggestions', methods=['GET'])
def get_suggestions():
    """
    Get conversation starter suggestions.
    
    Returns:
        JSON with list of suggested questions
    """
    suggestions = [
        "How do I sign the letter A?",
        "What are some common phrases?",
        "Give me tips for learning",
        "How does the detection work?",
        "How do I sign numbers?",
        "I'm finding this difficult",
        "What features does this app have?"
    ]
    
    return jsonify({
        'success': True,
        'suggestions': suggestions
    })


@chatbot_bp.route('/hint/<sign>', methods=['GET'])
def get_sign_hint(sign):
    """
    Get a hint for performing a specific sign.
    
    Args:
        sign: The sign to get a hint for
    
    Returns:
        JSON with hint text
    """
    try:
        from ml import SignClassifier
        classifier = SignClassifier()
        hint = classifier._get_hint(sign)
        
        return jsonify({
            'success': True,
            'sign': sign,
            'hint': hint
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get hint: {str(e)}'
        }), 500
