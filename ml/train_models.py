"""
Train Sign Language Classification Models
Trains SVM and Random Forest models on collected landmark data
"""

import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from preprocessor import LandmarkPreprocessor
import json

class ModelTrainer:
    """
    Trains and evaluates sign language classification models.
    """
    
    def __init__(self, data_path='../data/training_data/dataset.npz'):
        """
        Initialize trainer.
        
        Args:
            data_path: Path to training data (.npz file)
        """
        self.data_path = data_path
        self.preprocessor = LandmarkPreprocessor()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        self.svm_model = None
        self.rf_model = None
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_data(self):
        """Load training data from file."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        print(f"Loading data from {self.data_path}...")
        data = np.load(self.data_path)
        
        X = data['X']  # Raw landmarks (N, 63)
        y = data['y']  # Labels (N,)
        
        print(f"Loaded {len(X)} samples with {len(np.unique(y))} unique labels")
        
        return X, y
    
    def preprocess_data(self, X, y):
        """
        Preprocess landmarks - using raw landmarks without feature extraction.
        
        Args:
            X: Raw landmark data (N, 63)
            y: Labels (N,)
        
        Returns:
            X_features: Raw landmarks (no additional features)
            y_encoded: Encoded labels
        """
        print("\nPreprocessing data...")
        
        # Use raw landmarks directly (63 features per sample)
        X_features = X
        y_filtered = y
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y_filtered)
        
        # Save label encoder
        os.makedirs('../data/trained_models', exist_ok=True)
        joblib.dump(self.label_encoder, '../data/trained_models/label_encoder.pkl')
        print(f"✓ Saved label encoder with {len(self.label_encoder.classes_)} classes")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_features)
        
        # Save scaler
        joblib.dump(self.scaler, '../data/trained_models/scaler.pkl')
        print(f"✓ Saved feature scaler")
        
        print(f"✓ Preprocessed {len(X_features)} samples")
        print(f"  Feature shape: {X_features.shape}")
        
        return X_scaled, y_encoded
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nData split:")
        print(f"  Training samples: {len(self.X_train)}")
        print(f"  Test samples: {len(self.X_test)}")
    
    def train_svm(self, use_grid_search=True):
        """
        Train SVM model for motion-based signs.
        
        Args:
            use_grid_search: Whether to use grid search for hyperparameters
        """
        print("\n" + "="*60)
        print("TRAINING SVM MODEL (Motion-based signs)")
        print("="*60)
        
        if use_grid_search:
            print("Running grid search for optimal hyperparameters...")
            
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'poly']
            }
            
            svm = SVC(random_state=42)
            grid_search = GridSearchCV(
                svm, param_grid, cv=5, scoring='accuracy', 
                n_jobs=-1, verbose=1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            self.svm_model = grid_search.best_estimator_
            print(f"\n✓ Best parameters: {grid_search.best_params_}")
            print(f"✓ Best cross-validation score: {grid_search.best_score_:.4f}")
        else:
            print("Training with default parameters...")
            self.svm_model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
            self.svm_model.fit(self.X_train, self.y_train)
        
        # Evaluate
        train_score = self.svm_model.score(self.X_train, self.y_train)
        test_score = self.svm_model.score(self.X_test, self.y_test)
        
        print(f"\n✓ Training accuracy: {train_score:.4f}")
        print(f"✓ Test accuracy: {test_score:.4f}")
        
        # Save model
        joblib.dump(self.svm_model, '../data/trained_models/svm_model.pkl')
        print(f"✓ Saved SVM model")
    
    def train_random_forest(self, use_grid_search=True):
        """
        Train Random Forest model for shape-based signs.
        
        Args:
            use_grid_search: Whether to use grid search for hyperparameters
        """
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST MODEL (Shape-based signs)")
        print("="*60)
        
        if use_grid_search:
            print("Running grid search for optimal hyperparameters...")
            
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, scoring='accuracy',
                n_jobs=-1, verbose=1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            self.rf_model = grid_search.best_estimator_
            print(f"\n✓ Best parameters: {grid_search.best_params_}")
            print(f"✓ Best cross-validation score: {grid_search.best_score_:.4f}")
        else:
            print("Training with default parameters...")
            self.rf_model = RandomForestClassifier(
                n_estimators=100, max_depth=20, random_state=42
            )
            self.rf_model.fit(self.X_train, self.y_train)
        
        # Evaluate
        train_score = self.rf_model.score(self.X_train, self.y_train)
        test_score = self.rf_model.score(self.X_test, self.y_test)
        
        print(f"\n✓ Training accuracy: {train_score:.4f}")
        print(f"✓ Test accuracy: {test_score:.4f}")
        
        # Save model
        joblib.dump(self.rf_model, '../data/trained_models/rf_model.pkl')
        print(f"✓ Saved Random Forest model")
    
    def evaluate_models(self):
        """Evaluate both models and print detailed metrics."""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Get unique labels in test set
        unique_test_labels = np.unique(self.y_test)
        target_names = self.label_encoder.inverse_transform(unique_test_labels)
        
        # SVM evaluation
        if self.svm_model:
            print("\nSVM Model:")
            y_pred_svm = self.svm_model.predict(self.X_test)
            print(classification_report(
                self.y_test, y_pred_svm,
                labels=unique_test_labels,
                target_names=target_names
            ))
        
        # Random Forest evaluation
        if self.rf_model:
            print("\nRandom Forest Model:")
            y_pred_rf = self.rf_model.predict(self.X_test)
            print(classification_report(
                self.y_test, y_pred_rf,
                labels=unique_test_labels,
                target_names=target_names
            ))
        
        # Ensemble evaluation
        if self.svm_model and self.rf_model:
            print("\nEnsemble (Average):")
            y_pred_ensemble = []
            
            for i in range(len(self.X_test)):
                svm_pred = self.svm_model.predict([self.X_test[i]])[0]
                rf_pred = self.rf_model.predict([self.X_test[i]])[0]
                
                # Use voting
                if svm_pred == rf_pred:
                    y_pred_ensemble.append(svm_pred)
                else:
                    # Use SVM prediction as tiebreaker
                    y_pred_ensemble.append(svm_pred)
            
            ensemble_accuracy = accuracy_score(self.y_test, y_pred_ensemble)
            print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
    
    def save_training_info(self):
        """Save training information and metadata."""
        info = {
            'training_date': str(np.datetime64('now')),
            'num_samples': len(self.X_train) + len(self.X_test),
            'num_classes': len(self.label_encoder.classes_),
            'classes': self.label_encoder.classes_.tolist(),
            'feature_dim': self.X_train.shape[1],
            'test_size': len(self.X_test),
            'svm_test_accuracy': float(self.svm_model.score(self.X_test, self.y_test)),
            'rf_test_accuracy': float(self.rf_model.score(self.X_test, self.y_test))
        }
        
        info_path = '../data/trained_models/training_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\n✓ Saved training info to {info_path}")


def main():
    """Main training function."""
    print("\n" + "="*60)
    print("SIGN LANGUAGE MODEL TRAINING")
    print("="*60)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Load data
    try:
        X, y = trainer.load_data()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run data_collector.py first to collect training data.")
        return
    
    # Preprocess
    X_processed, y_encoded = trainer.preprocess_data(X, y)
    
    # Split data
    trainer.split_data(X_processed, y_encoded)
    
    # Train models
    use_grid_search = input("\nUse grid search for hyperparameters? (y/n) [slower but better]: ").strip().lower() == 'y'
    
    trainer.train_svm(use_grid_search=use_grid_search)
    trainer.train_random_forest(use_grid_search=use_grid_search)
    
    # Evaluate
    trainer.evaluate_models()
    
    # Save info
    trainer.save_training_info()
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60)
    print("\nTrained models saved to: ../data/trained_models/")
    print("  - svm_model.pkl")
    print("  - rf_model.pkl")
    print("  - scaler.pkl")
    print("  - label_encoder.pkl")
    print("\nThe application will automatically use these models.")


if __name__ == '__main__':
    main()
