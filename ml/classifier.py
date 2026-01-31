"""
Sign language classifier module.
Combines SVM and Random Forest models for sign classification.
"""

import numpy as np
import os
import joblib
from typing import Dict, List, Optional, Tuple
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from .preprocessor import LandmarkPreprocessor


class SignClassifier:
    """
    Sign language classifier using ensemble of ML models.
    
    Uses SVM for motion-based signs (dynamic gestures) and
    Random Forest for shape-based signs (static hand shapes).
    
    Attributes:
        svm_model: SVM classifier for motion signs
        rf_model: Random Forest classifier for shape signs
        scaler: Feature scaler for normalization
        preprocessor: Landmark preprocessor
    """
    
    # Sign categories with their characteristics
    MOTION_SIGNS = ['J', 'Z', 'hello', 'thank you', 'please', 'sorry', 'help']
    SHAPE_SIGNS = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O',
        'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'yes', 'no', 'love', 'friend', 'family'
    ]
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the sign classifier.
        
        Args:
            model_path: Path to saved models directory
        """
        if model_path is None:
            # Get absolute path relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, '..', 'data', 'trained_models')
        
        self.model_path = model_path
        self.preprocessor = LandmarkPreprocessor()
        
        # Initialize models
        self.svm_model = None
        self.rf_model = None
        self.scaler = StandardScaler()
        self.label_encoder = None
        
        # Try to load pre-trained models first
        if self._load_trained_models():
            print("✓ Using trained models")
        else:
            # Fall back to synthetic data
            print("⚠ Using synthetic models - train with real data for better accuracy")
            self._initialize_models()
        
        # Previous landmarks for motion detection
        self.previous_landmarks = None
        self.motion_buffer = []
    
    def _load_trained_models(self) -> bool:
        """
        Load user-trained models from disk.
        Returns True if successful, False otherwise.
        """
        svm_path = os.path.join(self.model_path, 'svm_model.pkl')
        rf_path = os.path.join(self.model_path, 'rf_model.pkl')
        scaler_path = os.path.join(self.model_path, 'scaler.pkl')
        encoder_path = os.path.join(self.model_path, 'label_encoder.pkl')
        
        if not all(os.path.exists(p) for p in [svm_path, rf_path, scaler_path, encoder_path]):
            return False
        
        try:
            self.svm_model = joblib.load(svm_path)
            self.rf_model = joblib.load(rf_path)
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(encoder_path)
            print(f"  Loaded {len(self.label_encoder.classes_)} sign classes")
            return True
        except Exception as e:
            print(f"Error loading trained models: {e}")
            return False
    
    def _initialize_models(self):
        """Initialize models with default parameters."""
        # SVM for motion-based classification
        self.svm_model = SVC(
            kernel='rbf',
            C=10.0,
            gamma='scale',
            probability=True,
            class_weight='balanced'
        )
        
        # Random Forest for shape-based classification
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        )
        
        # Train with sample data for basic functionality
        self._train_with_sample_data()
    
    def _train_with_sample_data(self):
        """
        Train models with synthetic sample data for demonstration.
        In production, replace with real training data.
        """
        # Generate synthetic training data
        np.random.seed(42)
        
        # Create sample features for each sign (shape-based)
        all_signs = self.SHAPE_SIGNS
        n_samples_per_sign = 20
        n_features = 63  # 21 landmarks * 3 coordinates
        
        X_shape = []
        y_shape = []
        
        for sign in all_signs:
            # Generate synthetic features for each sign
            # In reality, these would come from actual hand landmark data
            base_pattern = np.random.randn(n_features) * 0.1
            for _ in range(n_samples_per_sign):
                sample = base_pattern + np.random.randn(n_features) * 0.05
                X_shape.append(sample)
                y_shape.append(sign)
        
        X_shape = np.array(X_shape)
        y_shape = np.array(y_shape)
        
        # Fit scaler and transform
        X_shape_scaled = self.scaler.fit_transform(X_shape)
        
        # Train Random Forest
        self.rf_model.fit(X_shape_scaled, y_shape)
        
        # For SVM, use same data (in production, use motion features)
        self.svm_model.fit(X_shape_scaled, y_shape)
        
        # Create label encoder for consistency
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(y_shape)
    
    def _load_models(self):
        """Load pre-trained models from disk (deprecated - use _load_trained_models)."""
        try:
            svm_path = os.path.join(self.model_path, 'svm_motion_model.joblib')
            rf_path = os.path.join(self.model_path, 'rf_shape_model.joblib')
            scaler_path = os.path.join(self.model_path, 'scaler.joblib')
            
            if os.path.exists(svm_path):
                self.svm_model = joblib.load(svm_path)
            if os.path.exists(rf_path):
                self.rf_model = joblib.load(rf_path)
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                
        except Exception as e:
            print(f"Error loading models: {e}")
            self._initialize_models()
    
    def save_models(self, path: str):
        """
        Save trained models to disk.
        
        Args:
            path: Directory path to save models
        """
        os.makedirs(path, exist_ok=True)
        
        joblib.dump(self.svm_model, os.path.join(path, 'svm_motion_model.joblib'))
        joblib.dump(self.rf_model, os.path.join(path, 'rf_shape_model.joblib'))
        joblib.dump(self.scaler, os.path.join(path, 'scaler.joblib'))
    
    def classify(
        self,
        landmarks: List[Dict],
        target_sign: Optional[str] = None
    ) -> Dict:
        """
        Classify the current hand sign.
        
        Args:
            landmarks: List of 21 landmark dictionaries
            target_sign: Expected sign (for validation mode)
            
        Returns:
            dict: Classification results with predictions and confidence
        """
        try:
            # Preprocess landmarks
            features = self.preprocessor.extract_features(landmarks)
            raw_features = features['raw'].reshape(1, -1)
            
            # Scale features
            scaled_features = self.scaler.transform(raw_features)
            
            # Determine if this might be a motion sign
            is_motion = self._detect_motion(landmarks)
            
            # Get predictions from both models
            rf_prediction, rf_confidence = self._classify_shape(scaled_features)
            
            if is_motion and hasattr(self, 'motion_scaler'):
                motion_features = self._extract_motion_features(landmarks)
                svm_prediction, svm_confidence = self._classify_motion(motion_features)
            else:
                svm_prediction, svm_confidence = None, 0.0
            
            # Combine predictions
            if is_motion and svm_confidence > rf_confidence:
                final_prediction = svm_prediction
                final_confidence = svm_confidence
                model_used = 'svm'
            else:
                final_prediction = rf_prediction
                final_confidence = rf_confidence
                model_used = 'random_forest'
            
            # Check if prediction matches target
            is_correct = None
            if target_sign:
                is_correct = final_prediction.lower() == target_sign.lower()
            
            # Store current landmarks for next frame
            self.previous_landmarks = landmarks
            
            # Convert all numpy types to Python native types for JSON serialization
            return {
                'success': True,
                'prediction': str(final_prediction),  # numpy.str_ → str
                'confidence': float(final_confidence),
                'model_used': model_used,
                'is_correct': bool(is_correct) if is_correct is not None else None,  # numpy.bool_ → bool
                'is_motion_sign': bool(is_motion),  # numpy.bool_ → bool
                'alternatives': self._get_alternatives(scaled_features)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'prediction': None,
                'confidence': 0.0
            }
    
    def _classify_shape(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Classify using Random Forest (shape-based).
        
        Args:
            features: Scaled feature array
            
        Returns:
            tuple: (prediction, confidence)
        """
        pred_index = self.rf_model.predict(features)[0]
        probabilities = self.rf_model.predict_proba(features)[0]
        confidence = float(np.max(probabilities))
        
        # Convert numeric prediction to label string
        if self.label_encoder is not None:
            prediction = self.label_encoder.inverse_transform([pred_index])[0]
            print(f"🔍 RF: index={pred_index} → label='{prediction}' (confidence={confidence:.2%})")
        else:
            # Fallback if no encoder (synthetic model)
            prediction = str(pred_index)
        
        return prediction, confidence
    
    def _classify_motion(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Classify using SVM (motion-based).
        
        Args:
            features: Motion feature array
            
        Returns:
            tuple: (prediction, confidence)
        """
        if self.svm_model is None:
            return None, 0.0
        
        scaled = self.motion_scaler.transform(features.reshape(1, -1))
        pred_index = self.svm_model.predict(scaled)[0]
        probabilities = self.svm_model.predict_proba(scaled)[0]
        confidence = float(np.max(probabilities))
        
        # Convert numeric prediction to label string
        if self.label_encoder is not None:
            prediction = self.label_encoder.inverse_transform([pred_index])[0]
        else:
            # Fallback if no encoder (synthetic model)
            prediction = str(pred_index)
        
        return prediction, confidence
    
    def _detect_motion(self, current_landmarks: List[Dict]) -> bool:
        """
        Detect if there's significant motion between frames.
        
        Args:
            current_landmarks: Current frame landmarks
            
        Returns:
            bool: True if motion detected
        """
        if self.previous_landmarks is None:
            return False
        
        motion = self.preprocessor.calculate_motion(
            current_landmarks,
            self.previous_landmarks
        )
        
        return motion['is_moving'] and motion['hand_speed'] > 0.05
    
    def _extract_motion_features(self, landmarks: List[Dict]) -> np.ndarray:
        """
        Extract motion-specific features.
        
        Args:
            landmarks: Current landmarks
            
        Returns:
            numpy.ndarray: Motion features
        """
        # Current position features
        current = self.preprocessor._to_numpy(landmarks)
        
        # Motion features (if available)
        if self.previous_landmarks:
            motion = self.preprocessor.calculate_motion(
                landmarks,
                self.previous_landmarks
            )
            motion_features = motion['displacement'].flatten()
        else:
            motion_features = np.zeros(63)
        
        # Combine current and motion features
        return np.concatenate([current, motion_features])
    
    def _get_alternatives(self, features: np.ndarray, top_k: int = 3) -> List[Dict]:
        """
        Get alternative predictions with confidence scores.
        
        Args:
            features: Feature array
            top_k: Number of alternatives to return
            
        Returns:
            list: Alternative predictions with scores
        """
        probabilities = self.rf_model.predict_proba(features)[0]
        classes = self.rf_model.classes_
        
        # Sort by probability
        sorted_indices = np.argsort(probabilities)[::-1]
        
        alternatives = []
        for i in sorted_indices[:top_k]:
            # Convert numpy types to Python native types for JSON serialization
            sign_label = str(classes[i]) if hasattr(classes[i], 'item') else classes[i]
            alternatives.append({
                'sign': sign_label,
                'confidence': float(probabilities[i])
            })
        
        return alternatives
    
    def validate_sign(self, landmarks: List[Dict], expected_sign: str) -> Dict:
        """
        Validate if the detected sign matches the expected sign.
        
        Args:
            landmarks: Hand landmarks
            expected_sign: The sign the user should be performing
            
        Returns:
            dict: Validation result with feedback
        """
        result = self.classify(landmarks, target_sign=expected_sign)
        
        # Handle classification failure
        if not result.get('success', False):
            result['is_correct'] = False
            result['feedback'] = {
                'status': 'error',
                'message': 'Could not classify the sign. Please try again.',
                'score': 0
            }
            return result
        
        # Generate feedback
        if result.get('is_correct', False):
            feedback = {
                'status': 'correct',
                'message': f'Great job! You correctly signed "{expected_sign}"',
                'score': min(result['confidence'] * 100, 100)
            }
        else:
            feedback = {
                'status': 'incorrect',
                'message': f'That looks like "{result["prediction"]}". Try signing "{expected_sign}" again.',
                'hint': self._get_hint(expected_sign),
                'score': 0
            }
        
        result['feedback'] = feedback
        return result
    
    def _get_hint(self, sign: str) -> str:
        """
        Get a hint for performing a specific sign.
        
        Args:
            sign: The sign to get a hint for
            
        Returns:
            str: Hint text
        """
        hints = {
            'A': 'Make a fist with your thumb resting on the side',
            'B': 'Hold your hand flat with fingers together and thumb tucked',
            'C': 'Curve your hand like you\'re holding a cup',
            'D': 'Touch your thumb to your middle, ring, and pinky fingers',
            'E': 'Curl all fingers down with thumb tucked under',
            'F': 'Touch your thumb and index finger, other fingers up',
            'G': 'Point your index finger to the side, thumb up',
            'H': 'Point index and middle fingers to the side',
            'I': 'Make a fist with pinky extended',
            'J': 'Make an I and trace a J shape',
            'K': 'Index and middle fingers up, thumb between them',
            'L': 'Make an L shape with thumb and index finger',
            'M': 'Tuck thumb under three fingers',
            'N': 'Tuck thumb under two fingers',
            'O': 'Touch all fingertips to thumb forming an O',
            'P': 'Like K but pointing down',
            'Q': 'Like G but pointing down',
            'R': 'Cross your index and middle fingers',
            'S': 'Make a fist with thumb over fingers',
            'T': 'Thumb between index and middle fingers',
            'U': 'Hold up index and middle fingers together',
            'V': 'Hold up index and middle fingers in a V',
            'W': 'Hold up index, middle, and ring fingers',
            'X': 'Hook your index finger',
            'Y': 'Extend thumb and pinky',
            'Z': 'Trace a Z shape with your index finger',
            '0': 'Form an O shape',
            '1': 'Hold up your index finger',
            '2': 'Hold up index and middle fingers (V shape)',
            '3': 'Hold up thumb, index, and middle fingers',
            '4': 'Hold up four fingers with thumb tucked',
            '5': 'Hold up all five fingers',
            'hello': 'Wave your open hand',
            'thank you': 'Touch your chin and move hand forward',
            'please': 'Rub your chest in a circular motion',
            'sorry': 'Make a fist and rub it on your chest',
            'yes': 'Nod your fist up and down',
            'no': 'Snap index and middle fingers to thumb',
        }
        
        return hints.get(sign.upper(), hints.get(sign.lower(), 'Focus on the hand shape shown'))
    
    def reset(self):
        """Reset classifier state."""
        self.previous_landmarks = None
        self.motion_buffer = []
        self.preprocessor.reset()
