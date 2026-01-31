"""
Simplified hand detector for MediaPipe 0.10.32+
Uses the newer task-based API
"""

import cv2
import numpy as np
from typing import List, Dict
import base64
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request


class HandDetector:
    """Hand detector using MediaPipe Hands (new task API)"""
    
    LANDMARK_NAMES = [
        'WRIST',
        'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
        'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
        'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
        'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
        'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
    ]
    
    def __init__(
        self,
        max_hands: int = 2,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.5
    ):
        """Initialize hand detector"""
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        
        # Download model if not exists - use absolute path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'hand_landmarker.task')
        
        if not os.path.exists(model_path):
            print("Downloading hand landmarker model...")
            url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'
            try:
                urllib.request.urlretrieve(url, model_path)
                print("✓ Model downloaded")
            except Exception as e:
                print(f"Error downloading model: {e}")
                print("Using OpenCV-based detection fallback")
                self.detector = None
                return
        
        # Create hand landmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
    
    def detect(self, image):
        """Detect hands in image"""
        if self.detector is None:
            return {'success': False, 'landmarks': [], 'error': 'Detector not initialized'}
        
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            # Detect hands
            detection_result = self.detector.detect(mp_image)
            
            # Convert results to our format
            landmarks_list = []
            
            if detection_result.hand_landmarks:
                for hand_landmarks in detection_result.hand_landmarks:
                    landmarks = []
                    for landmark in hand_landmarks:
                        landmarks.append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z
                        })
                    landmarks_list.append(landmarks)
            
            return {
                'success': True,
                'landmarks': landmarks_list,
                'hands_detected': len(landmarks_list)
            }
            
        except Exception as e:
            return {'success': False, 'landmarks': [], 'error': str(e)}
    
    def detect_from_base64(self, image_data: str) -> Dict:
        """Detect hands from base64 encoded image"""
        try:
            if 'base64,' in image_data:
                image_data = image_data.split('base64,')[1]
            
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return {'success': False, 'error': 'Failed to decode image'}
            
            return self.detect(frame)
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_landmarks_array(self, landmarks):
        """Convert landmarks to numpy array"""
        if not landmarks:
            return None
        
        arr = []
        for lm in landmarks:
            arr.extend([lm['x'], lm['y'], lm['z']])
        return np.array(arr)
