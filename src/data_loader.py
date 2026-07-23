# src/data_loader.py
"""
Data Loader Module - Criminal Face Dataset
Unit I Concept: Training and Test Datasets
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class CriminalFaceDataset:
    """
    Criminal Face Dataset Loader and Preprocessor
    Implements Training and Test Datasets concept from Unit I
    """
    
    def __init__(self, data_path='data/criminal_faces', img_size=(128, 128)):
        self.data_path = data_path
        self.img_size = img_size
        self.X = None
        self.y = None
        self.class_names = ['non_criminal', 'criminal']
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        logger.info(f"Initialized CriminalFaceDataset with data_path: {data_path}")
    
    def load_dataset(self):
        """Load criminal face images from directory"""
        features = []
        labels = []
        
        logger.info(f"Loading dataset from: {self.data_path}")
        
        for class_id, class_name in enumerate(self.class_names):
            class_path = os.path.join(self.data_path, class_name)
            
            if not os.path.exists(class_path):
                logger.warning(f"Directory not found: {class_path}")
                logger.info("Creating sample data for demonstration...")
                return self._create_sample_data()
            
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            for img_name in image_files:
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    img = cv2.resize(img, self.img_size)
                    img = img / 255.0
                    features.append(img.flatten())
                    labels.append(class_id)
        
        if not features:
            logger.warning("No images found. Creating sample data...")
            return self._create_sample_data()
        
        self.X = np.array(features, dtype=np.float32)
        self.y = np.array(labels, dtype=np.int32)
        
        logger.info(f"✅ Dataset loaded: {len(self.X)} samples, {self.X.shape[1]} features")
        
        return self.X, self.y
    
    def _create_sample_data(self):
        """Create synthetic sample data for testing"""
        logger.info("Generating synthetic criminal face data...")
        
        np.random.seed(42)
        n_samples = 200
        n_features = self.img_size[0] * self.img_size[1]
        
        self.X = np.random.randn(n_samples, n_features) * 0.5 + 0.5
        self.X = np.clip(self.X, 0, 1)
        self.y = np.random.randint(0, 2, n_samples)
        
        logger.info(f"✅ Sample data created: {n_samples} samples")
        
        return self.X, self.y
    
    def split_train_test(self, test_size=0.2, random_state=42):
        """Split data into training and test sets"""
        if self.X is None or self.y is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        logger.info(f"✅ Split complete: {len(self.X_train)} train, {len(self.X_test)} test")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_data_summary(self):
        """Get dataset summary"""
        if self.X is None:
            return "Dataset not loaded"
        
        print("\n" + "="*70)
        print("📊 DATASET SUMMARY REPORT")
        print("="*70)
        print(f"  Total samples: {len(self.X)}")
        print(f"  Feature dimension: {self.X.shape[1]}")
        print(f"  Image size: {self.img_size}")
        print(f"  Classes: {self.class_names}")
        print(f"  Class distribution: {np.bincount(self.y)}")
        
        if self.X_train is not None:
            print(f"  Training samples: {len(self.X_train)}")
            print(f"  Test samples: {len(self.X_test)}")
