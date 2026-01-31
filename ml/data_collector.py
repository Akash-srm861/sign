"""
Data Collection Script for Sign Language Dataset
Captures hand landmarks from webcam and saves them for training
"""

import cv2
import numpy as np
import json
import os
from datetime import datetime
from hand_detector import HandDetector

class DataCollector:
    """
    Collects hand landmark data for training sign language models.
    Saves data in JSON format with labels.
    """
    
    def __init__(self, output_dir='../data/training_data'):
        """
        Initialize data collector.
        
        Args:
            output_dir: Directory to save collected data
        """
        self.output_dir = output_dir
        self.detector = HandDetector()
        self.current_sign = None
        self.collected_samples = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load existing data if available
        self.load_existing_data()
    
    def load_existing_data(self):
        """Load previously collected data."""
        data_file = os.path.join(self.output_dir, 'collected_data.json')
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                self.collected_samples = json.load(f)
            print(f"Loaded {len(self.collected_samples)} existing samples")
    
    def save_data(self):
        """Save collected data to JSON file."""
        data_file = os.path.join(self.output_dir, 'collected_data.json')
        with open(data_file, 'w') as f:
            json.dump(self.collected_samples, f, indent=2)
        print(f"Saved {len(self.collected_samples)} samples to {data_file}")
    
    def collect_samples(self, sign_name, category, num_samples=50):
        """
        Collect samples for a specific sign using webcam.
        
        Args:
            sign_name: Name of the sign (e.g., 'A', '1', 'hello')
            category: Category (alphabets, numbers, words)
            num_samples: Number of samples to collect
        """
        self.current_sign = sign_name
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot open webcam")
            return
        
        print(f"\n{'='*60}")
        print(f"Collecting data for sign: '{sign_name}' (Category: {category})")
        print(f"Target samples: {num_samples}")
        print(f"{'='*60}")
        print("\nInstructions:")
        print("- Position your hand in frame")
        print("- Press SPACE to capture sample")
        print("- Press 'q' to finish early")
        print("- Press 'r' to restart collection for this sign")
        print("\n")
        
        samples_collected = 0
        
        while samples_collected < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for mirror view
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            # Detect hands
            result = self.detector.detect(frame)
            
            # Draw landmarks on display frame
            if result['landmarks']:
                for hand_landmarks in result['landmarks']:
                    # Draw all landmarks
                    for landmark in hand_landmarks:
                        x = int(landmark['x'] * frame.shape[1])
                        y = int(landmark['y'] * frame.shape[0])
                        cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
                    
                    # Draw connections
                    connections = [
                        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                        (5, 9), (9, 13), (13, 17)  # Palm
                    ]
                    for start, end in connections:
                        start_point = (int(hand_landmarks[start]['x'] * frame.shape[1]),
                                     int(hand_landmarks[start]['y'] * frame.shape[0]))
                        end_point = (int(hand_landmarks[end]['x'] * frame.shape[1]),
                                   int(hand_landmarks[end]['y'] * frame.shape[0]))
                        cv2.line(display_frame, start_point, end_point, (255, 0, 0), 2)
            
            # Display info
            cv2.putText(display_frame, f"Sign: {sign_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Samples: {samples_collected}/{num_samples}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if result['landmarks']:
                cv2.putText(display_frame, "Hand detected - Press SPACE", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "No hand detected", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Data Collection', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Capture sample
            if key == ord(' ') and result['landmarks']:
                # Save all detected hands
                for landmarks in result['landmarks']:
                    sample = {
                        'sign_name': sign_name,
                        'category': category,
                        'landmarks': landmarks,
                        'timestamp': datetime.now().isoformat(),
                        'hand_count': len(result['landmarks'])
                    }
                    self.collected_samples.append(sample)
                    samples_collected += 1
                    print(f"✓ Captured sample {samples_collected}/{num_samples}")
                    
                    if samples_collected >= num_samples:
                        break
            
            # Restart collection
            elif key == ord('r'):
                # Remove samples for this sign
                self.collected_samples = [
                    s for s in self.collected_samples 
                    if s['sign_name'] != sign_name
                ]
                samples_collected = 0
                print(f"\nRestarting collection for '{sign_name}'...")
            
            # Quit
            elif key == ord('q'):
                print(f"\nFinished early with {samples_collected} samples")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Auto-save after each sign
        self.save_data()
        print(f"\n✓ Completed collection for '{sign_name}': {samples_collected} samples\n")
    
    def get_statistics(self):
        """Print statistics about collected data."""
        if not self.collected_samples:
            print("No data collected yet")
            return
        
        # Count by category and sign
        categories = {}
        for sample in self.collected_samples:
            cat = sample['category']
            sign = sample['sign_name']
            
            if cat not in categories:
                categories[cat] = {}
            if sign not in categories[cat]:
                categories[cat][sign] = 0
            categories[cat][sign] += 1
        
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        print(f"Total samples: {len(self.collected_samples)}")
        print(f"\nBreakdown by category:")
        
        for cat, signs in categories.items():
            print(f"\n{cat.upper()}:")
            for sign, count in sorted(signs.items()):
                print(f"  {sign}: {count} samples")
        print("\n" + "="*60)
    
    def export_for_training(self, output_file='../data/training_data/dataset.npz'):
        """
        Export collected data in format suitable for training.
        
        Args:
            output_file: Path to save numpy arrays
        """
        if not self.collected_samples:
            print("No data to export")
            return
        
        # Convert to numpy arrays
        X = []  # Features (landmarks)
        y = []  # Labels
        
        for sample in self.collected_samples:
            # Flatten landmarks to 1D array (21 landmarks x 3 coordinates = 63 features)
            landmarks_flat = []
            for landmark in sample['landmarks']:
                landmarks_flat.extend([landmark['x'], landmark['y'], landmark['z']])
            
            X.append(landmarks_flat)
            y.append(sample['sign_name'])
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        
        # Save as compressed numpy file
        np.savez_compressed(output_file, X=X, y=y)
        print(f"\n✓ Exported {len(X)} samples to {output_file}")
        print(f"  Features shape: {X.shape}")
        print(f"  Unique labels: {len(np.unique(y))}")


def main():
    """
    Main function to run data collection.
    Customize this based on your needs.
    """
    collector = DataCollector()
    
    print("\n" + "="*60)
    print("SIGN LANGUAGE DATA COLLECTION TOOL")
    print("="*60)
    
    # Example: Collect ASL alphabet
    print("\nCollecting ASL Alphabet (A-Z)")
    print("="*60)
    
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    for letter in alphabet:
        response = input(f"\nCollect data for letter '{letter}'? (y/n/q to quit): ").strip().lower()
        
        if response == 'q':
            break
        elif response == 'y':
            collector.collect_samples(
                sign_name=letter,
                category='alphabets',
                num_samples=50  # Adjust number of samples
            )
    
    # Example: Collect numbers
    print("\n\nCollecting ASL Numbers (0-9)")
    print("="*60)
    
    for number in range(10):
        response = input(f"\nCollect data for number '{number}'? (y/n/q to quit): ").strip().lower()
        
        if response == 'q':
            break
        elif response == 'y':
            collector.collect_samples(
                sign_name=str(number),
                category='numbers',
                num_samples=50
            )
    
    # Show statistics
    collector.get_statistics()
    
    # Export for training
    response = input("\nExport data for training? (y/n): ").strip().lower()
    if response == 'y':
        collector.export_for_training()
    
    print("\n✓ Data collection complete!")
    print("Next step: Run train_models.py to train classifiers")


if __name__ == '__main__':
    main()
