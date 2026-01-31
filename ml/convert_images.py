"""
Image Dataset to Landmarks Converter
Processes ASL image dataset and extracts MediaPipe hand landmarks
"""

import cv2
import numpy as np
import os
import json
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(__file__))
from hand_detector_new import HandDetector

class ImageDatasetConverter:
    """
    Converts image-based ASL dataset to MediaPipe landmarks.
    """
    
    def __init__(self, dataset_path, output_path='../data/training_data'):
        """
        Initialize converter.
        
        Args:
            dataset_path: Path to image dataset (folder with a/, b/, c/, etc.)
            output_path: Path to save processed landmarks
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.detector = HandDetector(
            max_hands=1,  # Process one hand per image
            detection_confidence=0.5,
            tracking_confidence=0.5
        )
        
        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)
        
        self.collected_samples = []
        self.stats = {
            'total_images': 0,
            'successful': 0,
            'failed': 0,
            'no_hand_detected': 0,
            'by_sign': {}
        }
    
    def get_sign_folders(self):
        """Get all sign folders (a, b, c, ..., 0, 1, 2, ...)."""
        folders = []
        
        # Check for alphabet folders (a-z)
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            folder = self.dataset_path / letter
            if folder.exists() and folder.is_dir():
                folders.append((letter.upper(), folder, 'alphabets'))
        
        # Check for number folders (0-9)
        for number in range(10):
            folder = self.dataset_path / str(number)
            if folder.exists() and folder.is_dir():
                folders.append((str(number), folder, 'numbers'))
        
        return folders
    
    def process_image(self, image_path, sign_name, category):
        """
        Process a single image and extract landmarks.
        
        Args:
            image_path: Path to image file
            sign_name: Sign label (e.g., 'A', 'B', '0')
            category: Category (alphabets, numbers)
        
        Returns:
            Sample dict or None if failed
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        # Detect hand landmarks
        result = self.detector.detect(image)
        
        if not result['landmarks'] or len(result['landmarks']) == 0:
            return None
        
        # Take first detected hand
        landmarks = result['landmarks'][0]
        
        # Create sample
        sample = {
            'sign_name': sign_name,
            'category': category,
            'landmarks': landmarks,
            'source_image': str(image_path.name),
            'timestamp': str(np.datetime64('now'))
        }
        
        return sample
    
    def process_folder(self, sign_name, folder_path, category):
        """
        Process all images in a folder.
        
        Args:
            sign_name: Sign label
            folder_path: Path to folder containing images
            category: Category (alphabets, numbers)
        
        Returns:
            List of successfully processed samples
        """
        samples = []
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [
            f for f in folder_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        print(f"\n  Processing {len(image_files)} images for '{sign_name}'...")
        
        successful = 0
        failed = 0
        
        for image_file in tqdm(image_files, desc=f"  {sign_name}", leave=False):
            self.stats['total_images'] += 1
            
            sample = self.process_image(image_file, sign_name, category)
            
            if sample:
                samples.append(sample)
                successful += 1
                self.stats['successful'] += 1
            else:
                failed += 1
                self.stats['failed'] += 1
                self.stats['no_hand_detected'] += 1
        
        # Update stats
        self.stats['by_sign'][sign_name] = {
            'successful': successful,
            'failed': failed,
            'total': len(image_files)
        }
        
        print(f"  ✓ {sign_name}: {successful}/{len(image_files)} images processed successfully")
        
        return samples
    
    def convert_dataset(self):
        """
        Convert entire image dataset to landmarks.
        """
        print("="*70)
        print("ASL IMAGE DATASET TO LANDMARKS CONVERTER")
        print("="*70)
        print(f"\nDataset path: {self.dataset_path}")
        print(f"Output path: {self.output_path}")
        
        # Get all sign folders
        sign_folders = self.get_sign_folders()
        
        if not sign_folders:
            print("\n✗ No sign folders found!")
            print(f"Expected folders like: a/, b/, c/, ..., 0/, 1/, 2/, ...")
            print(f"in: {self.dataset_path}")
            return
        
        print(f"\nFound {len(sign_folders)} sign folders")
        print(f"Signs: {', '.join([s[0] for s in sign_folders[:10]])}...")
        print("\nStarting conversion...\n")
        
        # Process each sign folder
        for sign_name, folder_path, category in sign_folders:
            samples = self.process_folder(sign_name, folder_path, category)
            self.collected_samples.extend(samples)
        
        # Save results
        self.save_data()
        self.print_statistics()
    
    def save_data(self):
        """Save collected landmarks to JSON and NPZ formats."""
        if not self.collected_samples:
            print("\n✗ No samples to save!")
            return
        
        # Save JSON format
        json_path = self.output_path / 'collected_data.json'
        with open(json_path, 'w') as f:
            json.dump(self.collected_samples, f, indent=2)
        print(f"\n✓ Saved JSON: {json_path}")
        
        # Convert to NumPy arrays
        X = []
        y = []
        
        for sample in self.collected_samples:
            # Flatten landmarks to 1D array (21 landmarks x 3 coords = 63 features)
            landmarks_flat = []
            for landmark in sample['landmarks']:
                landmarks_flat.extend([landmark['x'], landmark['y'], landmark['z']])
            
            X.append(landmarks_flat)
            y.append(sample['sign_name'])
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        
        # Save NPZ format
        npz_path = self.output_path / 'dataset.npz'
        np.savez_compressed(npz_path, X=X, y=y)
        print(f"✓ Saved NPZ: {npz_path}")
        print(f"  Shape: X={X.shape}, y={y.shape}")
    
    def print_statistics(self):
        """Print conversion statistics."""
        print("\n" + "="*70)
        print("CONVERSION STATISTICS")
        print("="*70)
        
        print(f"\nTotal images processed: {self.stats['total_images']}")
        print(f"  ✓ Successful: {self.stats['successful']} ({self.stats['successful']/max(self.stats['total_images'],1)*100:.1f}%)")
        print(f"  ✗ Failed: {self.stats['failed']} ({self.stats['failed']/max(self.stats['total_images'],1)*100:.1f}%)")
        print(f"    - No hand detected: {self.stats['no_hand_detected']}")
        
        print("\nSamples per sign:")
        for sign, counts in sorted(self.stats['by_sign'].items()):
            success_rate = counts['successful'] / max(counts['total'], 1) * 100
            print(f"  {sign}: {counts['successful']}/{counts['total']} ({success_rate:.1f}%)")
        
        # Recommendations
        print("\n" + "="*70)
        print("RECOMMENDATIONS")
        print("="*70)
        
        min_samples = min([counts['successful'] for counts in self.stats['by_sign'].values()])
        avg_samples = np.mean([counts['successful'] for counts in self.stats['by_sign'].values()])
        
        print(f"\nMinimum samples per sign: {min_samples}")
        print(f"Average samples per sign: {avg_samples:.1f}")
        
        if min_samples < 30:
            print("\n⚠ Warning: Some signs have fewer than 30 samples")
            print("   Recommendation: Collect more samples or use data augmentation")
        elif min_samples < 50:
            print("\n✓ Good: Most signs have 30+ samples")
            print("   Recommendation: 50+ samples per sign would be better")
        else:
            print("\n✓ Excellent: All signs have 50+ samples")
            print("   Your dataset is ready for training!")
        
        if self.stats['failed'] > self.stats['successful'] * 0.2:
            print(f"\n⚠ Warning: High failure rate ({self.stats['failed']/max(self.stats['total_images'],1)*100:.1f}%)")
            print("   Check image quality and hand visibility")
        
        print("\n" + "="*70)
        print("✓ CONVERSION COMPLETE")
        print("="*70)
        print("\nNext steps:")
        print("1. Review the statistics above")
        print("2. Run: python train_models.py")
        print("3. Or use: train_models.bat")


def main():
    """Main conversion function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert ASL image dataset to landmarks')
    parser.add_argument(
        '--dataset',
        type=str,
        default='../../asl_dataset',
        help='Path to image dataset (default: ../../asl_dataset)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../data/training_data',
        help='Output path for landmarks (default: ../data/training_data)'
    )
    
    args = parser.parse_args()
    
    # Create converter
    converter = ImageDatasetConverter(
        dataset_path=args.dataset,
        output_path=args.output
    )
    
    # Convert dataset
    converter.convert_dataset()


if __name__ == '__main__':
    main()
