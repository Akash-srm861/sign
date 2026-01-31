"""
Landmark preprocessing module.
Handles normalization, noise removal, and feature extraction for ML models.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque
import math


class LandmarkPreprocessor:
    """
    Preprocessor for hand landmark data.
    
    Performs normalization, noise filtering, and feature extraction
    to prepare landmark data for ML classification.
    
    Attributes:
        smoothing_window: Number of frames for temporal smoothing
        noise_threshold: Minimum movement to consider (removes jitter)
        normalize: Whether to apply normalization
    """
    
    def __init__(
        self,
        smoothing_window: int = 5,
        noise_threshold: float = 0.02,
        normalize: bool = True
    ):
        """
        Initialize the preprocessor.
        
        Args:
            smoothing_window: Window size for temporal smoothing
            noise_threshold: Threshold for noise filtering
            normalize: Whether to normalize landmarks
        """
        self.smoothing_window = smoothing_window
        self.noise_threshold = noise_threshold
        self.normalize = normalize
        
        # Buffer for temporal smoothing
        self.landmark_buffer = deque(maxlen=smoothing_window)
        
        # Store reference frame for motion detection
        self.reference_landmarks = None
    
    def preprocess(self, landmarks: List[Dict]) -> np.ndarray:
        """
        Main preprocessing pipeline for landmarks.
        
        Args:
            landmarks: List of 21 landmark dictionaries with x, y, z
            
        Returns:
            numpy.ndarray: Preprocessed feature vector
        """
        # Convert to numpy array
        raw_array = self._to_numpy(landmarks)
        
        # Apply smoothing
        smoothed = self._temporal_smooth(raw_array)
        
        # Apply noise removal
        filtered = self._remove_noise(smoothed)
        
        # Normalize if enabled
        if self.normalize:
            normalized = self._normalize_landmarks(filtered)
        else:
            normalized = filtered
        
        return normalized
    
    def extract_features(self, landmarks: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive features for classification.
        
        Args:
            landmarks: List of 21 landmark dictionaries
            
        Returns:
            dict: Dictionary containing different feature types
        """
        # Preprocess landmarks
        preprocessed = self.preprocess(landmarks)
        
        # Reshape for feature extraction (21 landmarks, 3 coords each)
        points = preprocessed.reshape(21, 3)
        
        # Extract various feature types
        features = {
            'raw': preprocessed,  # Shape: (63,)
            'distances': self._extract_distances(points),  # Pairwise distances
            'angles': self._extract_angles(points),  # Joint angles
            'relative': self._extract_relative_positions(points),  # Relative to wrist
            'fingertip_distances': self._extract_fingertip_features(points),
            'hand_geometry': self._extract_hand_geometry(points)
        }
        
        # Combine all features for ML
        features['combined'] = np.concatenate([
            features['relative'].flatten(),
            features['distances'],
            features['angles'],
            features['fingertip_distances'],
            features['hand_geometry']
        ])
        
        return features
    
    def _to_numpy(self, landmarks: List[Dict]) -> np.ndarray:
        """
        Convert landmark list to numpy array.
        
        Args:
            landmarks: List of landmark dictionaries
            
        Returns:
            numpy.ndarray: Array of shape (63,)
        """
        arr = []
        for lm in landmarks:
            arr.extend([lm['x'], lm['y'], lm['z']])
        return np.array(arr, dtype=np.float32)
    
    def _temporal_smooth(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Apply temporal smoothing using moving average.
        
        Args:
            landmarks: Current frame landmarks
            
        Returns:
            numpy.ndarray: Smoothed landmarks
        """
        self.landmark_buffer.append(landmarks)
        
        if len(self.landmark_buffer) < 2:
            return landmarks
        
        # Weighted moving average (recent frames weighted more)
        weights = np.linspace(0.5, 1.0, len(self.landmark_buffer))
        weights /= weights.sum()
        
        smoothed = np.zeros_like(landmarks)
        for i, frame in enumerate(self.landmark_buffer):
            smoothed += weights[i] * frame
        
        return smoothed
    
    def _remove_noise(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Remove noise by filtering small movements.
        
        Args:
            landmarks: Current landmarks
            
        Returns:
            numpy.ndarray: Noise-filtered landmarks
        """
        if self.reference_landmarks is None:
            self.reference_landmarks = landmarks.copy()
            return landmarks
        
        # Calculate movement from reference
        diff = np.abs(landmarks - self.reference_landmarks)
        
        # Create mask for significant movements
        mask = diff > self.noise_threshold
        
        # Apply filtered update
        filtered = self.reference_landmarks.copy()
        filtered[mask] = landmarks[mask]
        
        # Update reference
        self.reference_landmarks = filtered.copy()
        
        return filtered
    
    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks to be scale and translation invariant.
        
        Centers landmarks on wrist and scales by hand size.
        
        Args:
            landmarks: Raw landmark array
            
        Returns:
            numpy.ndarray: Normalized landmarks
        """
        # Reshape to (21, 3)
        points = landmarks.reshape(21, 3)
        
        # Center on wrist (landmark 0)
        wrist = points[0].copy()
        centered = points - wrist
        
        # Calculate hand size (wrist to middle finger MCP)
        hand_size = np.linalg.norm(centered[9])  # Middle finger MCP
        
        # Avoid division by zero
        if hand_size < 1e-6:
            hand_size = 1.0
        
        # Scale to unit size
        normalized = centered / hand_size
        
        return normalized.flatten()
    
    def _extract_distances(self, points: np.ndarray) -> np.ndarray:
        """
        Extract pairwise distances between key landmarks.
        
        Args:
            points: Landmark points (21, 3)
            
        Returns:
            numpy.ndarray: Distance features
        """
        # Key landmark pairs for distance calculation
        pairs = [
            (4, 8),   # Thumb tip to index tip
            (8, 12),  # Index tip to middle tip
            (12, 16), # Middle tip to ring tip
            (16, 20), # Ring tip to pinky tip
            (4, 20),  # Thumb tip to pinky tip
            (0, 4),   # Wrist to thumb tip
            (0, 8),   # Wrist to index tip
            (0, 12),  # Wrist to middle tip
            (0, 20),  # Wrist to pinky tip
            (4, 12),  # Thumb to middle
            (5, 17),  # Index MCP to pinky MCP (palm width)
        ]
        
        distances = []
        for i, j in pairs:
            dist = np.linalg.norm(points[i] - points[j])
            distances.append(dist)
        
        return np.array(distances, dtype=np.float32)
    
    def _extract_angles(self, points: np.ndarray) -> np.ndarray:
        """
        Extract joint angles for each finger.
        
        Args:
            points: Landmark points (21, 3)
            
        Returns:
            numpy.ndarray: Angle features (in radians)
        """
        angles = []
        
        # Finger joint triplets (for angle calculation)
        finger_joints = [
            # Thumb
            [(1, 2, 3), (2, 3, 4)],
            # Index
            [(5, 6, 7), (6, 7, 8)],
            # Middle
            [(9, 10, 11), (10, 11, 12)],
            # Ring
            [(13, 14, 15), (14, 15, 16)],
            # Pinky
            [(17, 18, 19), (18, 19, 20)]
        ]
        
        for finger in finger_joints:
            for a, b, c in finger:
                angle = self._calculate_angle(points[a], points[b], points[c])
                angles.append(angle)
        
        return np.array(angles, dtype=np.float32)
    
    def _calculate_angle(
        self,
        point1: np.ndarray,
        point2: np.ndarray,
        point3: np.ndarray
    ) -> float:
        """
        Calculate angle at point2 formed by point1-point2-point3.
        
        Args:
            point1, point2, point3: 3D points
            
        Returns:
            float: Angle in radians
        """
        v1 = point1 - point2
        v2 = point3 - point2
        
        # Normalize vectors
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm < 1e-6 or v2_norm < 1e-6:
            return 0.0
        
        v1 = v1 / v1_norm
        v2 = v2 / v2_norm
        
        # Calculate angle
        cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle = math.acos(cos_angle)
        
        return angle
    
    def _extract_relative_positions(self, points: np.ndarray) -> np.ndarray:
        """
        Extract positions relative to wrist.
        
        Args:
            points: Landmark points (21, 3)
            
        Returns:
            numpy.ndarray: Relative position features (20, 3) - excluding wrist
        """
        wrist = points[0]
        relative = points[1:] - wrist
        
        # Normalize by hand span
        hand_span = np.max(np.linalg.norm(relative, axis=1))
        if hand_span > 1e-6:
            relative = relative / hand_span
        
        return relative
    
    def _extract_fingertip_features(self, points: np.ndarray) -> np.ndarray:
        """
        Extract features specifically from fingertips.
        
        Args:
            points: Landmark points (21, 3)
            
        Returns:
            numpy.ndarray: Fingertip features
        """
        tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        
        features = []
        
        # Distances between consecutive fingertips
        for i in range(len(tips) - 1):
            dist = np.linalg.norm(points[tips[i]] - points[tips[i+1]])
            features.append(dist)
        
        # Heights of fingertips (y-coordinate relative to wrist)
        for tip in tips:
            height = points[0][1] - points[tip][1]  # Lower y = higher position
            features.append(height)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_hand_geometry(self, points: np.ndarray) -> np.ndarray:
        """
        Extract overall hand geometry features.
        
        Args:
            points: Landmark points (21, 3)
            
        Returns:
            numpy.ndarray: Hand geometry features
        """
        features = []
        
        # Palm center (average of palm landmarks)
        palm_indices = [0, 5, 9, 13, 17]
        palm_center = np.mean(points[palm_indices], axis=0)
        
        # Hand bounding box dimensions
        bbox_min = np.min(points, axis=0)
        bbox_max = np.max(points, axis=0)
        bbox_size = bbox_max - bbox_min
        features.extend(bbox_size)
        
        # Aspect ratio
        if bbox_size[0] > 1e-6:
            aspect_ratio = bbox_size[1] / bbox_size[0]
        else:
            aspect_ratio = 1.0
        features.append(aspect_ratio)
        
        # Palm to fingertip distances
        for tip in [4, 8, 12, 16, 20]:
            dist = np.linalg.norm(points[tip] - palm_center)
            features.append(dist)
        
        return np.array(features, dtype=np.float32)
    
    def calculate_motion(self, current: List[Dict], previous: List[Dict]) -> Dict:
        """
        Calculate motion between two frames.
        
        Args:
            current: Current frame landmarks
            previous: Previous frame landmarks
            
        Returns:
            dict: Motion features including velocity and direction
        """
        curr_arr = self._to_numpy(current).reshape(21, 3)
        prev_arr = self._to_numpy(previous).reshape(21, 3)
        
        # Calculate displacement
        displacement = curr_arr - prev_arr
        
        # Calculate velocity (magnitude of displacement)
        velocity = np.linalg.norm(displacement, axis=1)
        
        # Overall hand movement (wrist displacement)
        hand_movement = displacement[0]
        hand_speed = np.linalg.norm(hand_movement)
        
        # Movement direction
        if hand_speed > 1e-6:
            direction = hand_movement / hand_speed
        else:
            direction = np.zeros(3)
        
        return {
            'displacement': displacement,
            'velocity': velocity,
            'hand_speed': hand_speed,
            'direction': direction,
            'is_moving': hand_speed > self.noise_threshold
        }
    
    def reset(self):
        """Reset the preprocessor state."""
        self.landmark_buffer.clear()
        self.reference_landmarks = None
