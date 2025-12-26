import cv2
import os
import numpy as np
from skimage.feature import hog
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.pipeline import make_pipeline as make_imb_pipeline
from imblearn.over_sampling import SMOTE
import joblib
import time

# Configuration
IMAGE_SIZE = (128, 128)
MAX_SAMPLES_PER_CLASS = 1500  # Balance class distribution

def extract_enhanced_features(img):
    """Enhanced feature extraction for emotion recognition"""
    # Histogram of Oriented Gradients (HOG)
    hog_features = hog(img, orientations=9, 
                      pixels_per_cell=(32, 32),
                      cells_per_block=(2, 2),
                      block_norm='L2-Hys')
    
    # Local Binary Patterns (LBP)
    lbp = cv2.LBP_create(radius=2, neighbors=16).compute(img)
    hist_lbp = cv2.calcHist([lbp], [0], None, [256], [0, 256]).flatten()
    
    # Edge features using Canny
    edges = cv2.Canny(img, 50, 150).flatten()
    
    return np.concatenate([
        hog_features,
        hist_lbp * (1/255.0),
        edges * (1/255.0)
    ])

def load_data_with_augmentation(data_dir):
    """Enhanced data loading with balanced augmentation"""
    images = []
    labels = []
    label_map = {}
    class_counts = Counter()
    current_label = 0
    total_images = 0
    start_time = time.time()

    # First pass to create label map
    for folder in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, folder)):
            label_map[folder] = current_label
            current_label += 1

    # Second pass for data loading
    for folder, class_id in label_map.items():
        folder_path = os.path.join(data_dir, folder)
        class_images = 0
        
        for img_name in os.listdir(folder_path)[:MAX_SAMPLES_PER_CLASS]:
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            img = cv2.resize(img, IMAGE_SIZE)
            
            # Original image features
            features = extract_enhanced_features(img)
            images.append(features)
            labels.append(class_id)
            class_images += 1
            total_images += 1
            
            # Augmentations
            # Horizontal flip
            flipped = cv2.flip(img, 1)
            images.append(extract_enhanced_features(flipped))
            labels.append(class_id)
            class_images += 1
            
            # Add noise
            noisy = cv2.addWeighted(img, 0.85, 
                                  np.random.normal(0, 15, img.shape).astype(np.uint8), 
                                  0.15, 0)
            images.append(extract_enhanced_features(noisy))
            labels.append(class_id)
            class_images += 1
            
            if class_images >= MAX_SAMPLES_PER_CLASS:
                break

    print(f"\nLoaded {total_images} samples across {len(label_map)} classes")
    print(f"Class distribution: {Counter(labels)}")
    print(f"Loading took {time.time() - start_time:.2f} seconds")
    return np.array(images), np.array(labels), label_map

def train_optimized_model():
    overall_start = time.time()
    print("Starting optimized training process...\n")

    # Load and prepare data
    X, y, label_map = load_data_with_augmentation('data/train')
    
    # Split data before processing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    # Create optimized pipeline
    pipeline = make_imb_pipeline(
        StandardScaler(),
        SMOTE(random_state=42, k_neighbors=5),
        PCA(n_components=0.98),
        VotingClassifier([
            ('gbc', GradientBoostingClassifier(
                n_estimators=300, 
                learning_rate=0.05,
                max_depth=7,
                subsample=0.8,
                random_state=42
            )),
            ('svm', SVC(
                C=5, 
                gamma='auto', 
                kernel='rbf',
                probability=True,
                random_state=42
            )),
            ('rf', RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                class_weight='balanced',
                random_state=42
            ))
        ], voting='soft', weights=[3, 2, 2])
    )

    # Train model
    print("\nTraining optimized ensemble model...")
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Evaluate
    print("\nModel Evaluation:")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=label_map.keys(), digits=4))

    # Save model
    os.makedirs("model", exist_ok=True)
    joblib.dump((pipeline, label_map), "model/emotion_model.pkl")
    
    print(f"\n✅ Model saved to 'model/emotion_model.pkl'")
    print(f"Training time: {train_time/60:.2f} minutes")
    print(f"Total runtime: {(time.time() - overall_start)/60:.2f} minutes")

if __name__ == "__main__":
    train_optimized_model()