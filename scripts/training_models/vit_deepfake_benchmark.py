import os
import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.imagenet_utils import preprocess_input

# If you haven't installed vit-keras: pip install vit-keras
# And make sure tensorflow-macos/tensorflow-metal are installed for macOS:
# /opt/anaconda3/bin/python -m pip install tensorflow-macos==2.15.0 tensorflow-metal vit-keras tensorflow-addons

from vit_keras import vit

# Define directory paths
train_dir = 'give the pathname to your directory'
val_dir = 'give the pathname to your directory'
test_dir = 'give the pathname to your directory'

IMAGE_SIZE  = 224
BATCH_SIZE  = 32
LR_INIT     = 1e-3
LR_FINE     = 1e-4
EPOCHS_INIT = 5
EPOCHS_FINE = 5
FREEZE_UPTO = -10   # unfreeze last 10 layers for fine-tuning

# Data Generators
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True,
    rotation_range=15
)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)
val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)
test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Model Setup
base_model = vit.vit_b16(
    image_size=IMAGE_SIZE,
    pretrained=True,
    include_top=False,
    pretrained_top=False
)

# Add custom head directly onto the 768-D embedding
x = base_model.output            # shape = (None, 768)
out = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=out)

# Phase 1: Initial Training
for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=LR_INIT),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n=== Starting Phase 1: Initial Training ===")
start = time.time()
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_INIT,
    verbose=1
)
print(f"Phase 1 took {(time.time() - start)/60:.2f} minutes\n")

# Phase 2: Fine-Tuning
for layer in base_model.layers[FREEZE_UPTO:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=LR_FINE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("=== Starting Phase 2: Fine-Tuning ===")
start = time.time()
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINE,
    verbose=1
)
print(f"Phase 2 took {(time.time() - start)/60:.2f} minutes\n")

# Save Model
model.save('vit_deepfake_detector.h5')
print("Saved trained model to vit_deepfake_detector.h5\n")

# Evaluation
print("=== Evaluating on Test Set ===")
pred_prob   = model.predict(test_gen)
pred_labels = (pred_prob > 0.5).astype(int).flatten()
true_labels = test_gen.classes

acc  = accuracy_score(true_labels, pred_labels)
prec = precision_score(true_labels, pred_labels)
rec  = recall_score(true_labels, pred_labels)
f1   = f1_score(true_labels, pred_labels)
auc  = roc_auc_score(true_labels, pred_prob)
cm   = confusion_matrix(true_labels, pred_labels)

print(f"Accuracy:          {acc:.4f}")
print(f"Precision:         {prec:.4f}")
print(f"Recall:            {rec:.4f}")
print(f"F1 Score:          {f1:.4f}")
print(f"ROC AUC:           {auc:.4f}")
print("Confusion Matrix:")
print(cm)