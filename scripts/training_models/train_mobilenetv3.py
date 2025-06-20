"""
MobileNetV3-Large deepfake detector training script.
Replace 'path/to/train', 'path/to/val', and 'path/to/test' with your data directories.
"""

import os
import platform
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Use legacy Adam optimizer on Apple Silicon to avoid Metal crash
if platform.system() == 'Darwin' and platform.processor() == 'arm':
    from tensorflow.keras.optimizers.legacy import Adam
else:
    from tensorflow.keras.optimizers import Adam

# Paths to data directories
train_dir = 'path/to/train'
val_dir   = 'path/to/val'
test_dir  = 'path/to/test'

# Hyperparameters
IMAGE_SIZE    = 224
BATCH_SIZE    = 32
LR_INIT       = 1e-3
LR_FINE       = 1e-4
EPOCHS_INIT   = 5
EPOCHS_FINE   = 5
UNFREEZE_LAST = 20

# Prepare data generators
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True,
    rotation_range=15
).flow_from_directory(
    train_dir,
    (IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
).flow_from_directory(
    val_dir,
    (IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
).flow_from_directory(
    test_dir,
    (IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Build MobileNetV3 model
inp = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_tensor=inp)
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
out = Dense(1, activation='sigmoid')(x)
model = Model(inputs=inp, outputs=out)

# Phase 1: Train classifier head
for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=LR_INIT),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
print("=== Phase 1: Training head ===")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_INIT,
    verbose=1
)

# Phase 2: Fine-tune last layers of backbone
for layer in base_model.layers[-UNFREEZE_LAST:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=LR_FINE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
print("=== Phase 2: Fine-tuning backbone ===")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINE,
    verbose=1
)

# Save the trained model
model.save('mobilenetv3_deepfake_detector_FF+C40.h5')
print("Model saved as 'mobilenetv3_deepfake_detector_FF+C40.h5'")

# Evaluate on test set
print("=== Test evaluation ===")
probs = model.predict(test_gen, verbose=1)
y_pred = (probs > 0.5).astype(int).flatten()
y_true = test_gen.classes

print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred):.4f}")
print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_true, y_pred):.4f}")
print(f"AUC      : {roc_auc_score(y_true, probs):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
