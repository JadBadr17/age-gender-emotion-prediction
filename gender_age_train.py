import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, applications, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
NUM_AGE_BINS = 10  # Reduced from 100 years to 10 bins
GENDER_CLASSES = 2

def parse_filename(filename):
    """Parse UTKFace filename into age and gender"""
    parts = filename.split('_')
    age = int(parts[0])
    gender = int(parts[1])
    return age, gender

def create_age_bins(age):
    """Convert age to bins with decade-based grouping"""
    return min(age // 10, NUM_AGE_BINS-1)

def load_and_preprocess_data(dataset_path):
    """Load images with multi-task labels (age + gender)"""
    images = []
    age_labels = []
    gender_labels = []
    
    for filename in os.listdir(dataset_path):
        if not filename.endswith('.jpg'):
            continue
            
        try:
            age, gender = parse_filename(filename)
            img_path = os.path.join(dataset_path, filename)
            
            # Load and preprocess image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = applications.mobilenet_v2.preprocess_input(img)
            
            # Create bins and labels
            age_bin = create_age_bins(age)
            
            images.append(img)
            age_labels.append(age_bin)
            gender_labels.append(gender)
        except:
            continue
    
    return np.array(images), (np.array(age_labels), np.array(gender_labels))

def create_multi_task_model():
    """Create multi-output model with shared backbone"""
    base_model = applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = True
    
    # Shared features
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Age branch (regression)
    age_output = layers.Dense(NUM_AGE_BINS, activation='softmax', name='age')(x)
    
    # Gender branch (classification)
    gender_output = layers.Dense(GENDER_CLASSES, activation='softmax', name='gender')(x)
    
    model = Model(inputs=base_model.input, outputs=[age_output, gender_output])
    
    model.compile(optimizer=Adam(LEARNING_RATE),
                  loss={'age': 'sparse_categorical_crossentropy',
                        'gender': 'sparse_categorical_crossentropy'},
                  metrics={'age': 'accuracy',
                           'gender': 'accuracy'},
                  loss_weights={'age': 0.7, 'gender': 0.3})
    
    return model

def data_augmentation():
    """Create data augmentation layer"""
    return tf.keras.Sequential([
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomFlip('horizontal'),
        layers.RandomContrast(0.2)
    ])

# Load and split data
images, (age_labels, gender_labels) = load_and_preprocess_data('UTKFace')
X_train, X_val, age_train, age_val, gender_train, gender_val = train_test_split(
    images, age_labels, gender_labels, test_size=0.2, stratify=gender_labels
)

# Create model
model = create_multi_task_model()
augmentation = data_augmentation()

# Create datasets
train_dataset = tf.data.Dataset.from_tensor_slices(
    (X_train, {'age': age_train, 'gender': gender_train})
).shuffle(1000).batch(BATCH_SIZE).map(
    lambda x, y: (augmentation(x, training=True), y), 
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (X_val, {'age': age_val, 'gender': gender_val})
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Callbacks
callbacks = [
    ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss'),
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=3)
]

# Train model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Evaluate
print("Final Evaluation:")
model.evaluate(val_dataset)

# Save final model
model.save('age_gender_classifier.h5')