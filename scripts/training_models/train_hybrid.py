"""
Hybrid Xception + ViT-B16 deepfake detector training script.
Replace 'path/to/train', 'path/to/val', and 'path/to/test' with your data directories.
"""

import os
import platform
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications.xception import Xception
from vit_keras import vit

# Use mixed precision for better performance
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# Use legacy Adam optimizer on Apple Silicon to avoid Metal deadlock
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

# Build hybrid model
inp = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

# Xception branch
cnn_base = Xception(weights='imagenet', include_top=False, input_tensor=inp)
cnn_feat = GlobalAveragePooling2D()(cnn_base.output)

# ViT branch
vit_backbone = vit.vit_b16(
    image_size=IMAGE_SIZE,
    pretrained=True,
    include_top=False,
    pretrained_top=False
)
vit_feat = vit_backbone(inp)

# Fusion and classification head
x = Concatenate()([cnn_feat, vit_feat])
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
out = Dense(1, activation='sigmoid', dtype='float32')(x)
model = Model(inputs=inp, outputs=out)

# Phase 1: Train classifier head
for layer in (*cnn_base.layers, *vit_backbone.layers):
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
for branch in (cnn_base, vit_backbone):
    for layer in branch.layers[-UNFREEZE_LAST:]:
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
model.save('hybrid_deepfake_detector_CelebDF.h5')
print("Model saved as 'hybrid_deepfake_detector_CelebDF.h5'")

# Evaluate on test set
print("=== Test evaluation ===")
probs = model.predict(test_gen, verbose=1)
y_pred = (probs > 0.5).astype(int).ravel()
y_true = test_gen.classes

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred):.4f}")
print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_true, y_pred):.4f}")
print(f"AUC      : {roc_auc_score(y_true, probs):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
