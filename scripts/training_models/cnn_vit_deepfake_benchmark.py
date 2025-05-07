import os, time, numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (Input, Dense, Dropout, GlobalAveragePooling2D,
                                     Concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications.xception import Xception
from vit_keras import vit      # pip install vit-keras

# Define directory paths
train_dir = 'give the pathname to your directory'
val_dir = 'give the pathname to your directory'
test_dir = 'give the pathname to your directory'

# Hyperparams
IMAGE_SIZE   = 224
BATCH_SIZE   = 32
LR_INIT      = 1e-3
LR_FINE      = 1e-4
EPOCHS_INIT  = 5
EPOCHS_FINE  = 5
UNFREEZE_LAST = 20          # layers to unfreeze during fine-tuning

# Data
train_gen = ImageDataGenerator(
                preprocessing_function=preprocess_input,   # <-- fixed name
                horizontal_flip=True, rotation_range=15
            ).flow_from_directory(
                train_dir, target_size=(IMAGE_SIZE, IMAGE_SIZE),
                batch_size=BATCH_SIZE, class_mode='binary')

val_gen = ImageDataGenerator(
                preprocessing_function=preprocess_input    # <-- fixed
            ).flow_from_directory(
                val_dir, target_size=(IMAGE_SIZE, IMAGE_SIZE),
                batch_size=BATCH_SIZE, class_mode='binary')

test_gen = ImageDataGenerator(
                preprocessing_function=preprocess_input    # <-- fixed
            ).flow_from_directory(
                test_dir, target_size=(IMAGE_SIZE, IMAGE_SIZE),
                batch_size=BATCH_SIZE, class_mode='binary', shuffle=False)

# Model
inp = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

# CNN branch (Xception)
cnn_base = Xception(weights='imagenet', include_top=False, input_tensor=inp)
cnn_feat = GlobalAveragePooling2D()(cnn_base.output)        # (None, 2048)

# ViT branch
vit_backbone = vit.vit_b16(image_size=IMAGE_SIZE,
                           pretrained=True,
                           include_top=False,
                           pretrained_top=False)
vit_feat = vit_backbone(inp)                                # (None, 768)

# Fuse & head
x = Concatenate()([cnn_feat, vit_feat])                     # (None, 2816)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
out = Dense(1, activation='sigmoid')(x)

model = Model(inp, out)

# Phase 1: Initial Training
for layer in (*cnn_base.layers, *vit_backbone.layers):
    layer.trainable = False

model.compile(Adam(LR_INIT), 'binary_crossentropy', metrics=['accuracy'])
print("\n=== Phase-1: head training ===")
model.fit(train_gen, validation_data=val_gen,
          epochs=EPOCHS_INIT, verbose=1)

# Phase 2: Fine-Tuning
for branch in (cnn_base, vit_backbone):
    for layer in branch.layers[-UNFREEZE_LAST:]:
        layer.trainable = True

model.compile(Adam(LR_FINE), 'binary_crossentropy', metrics=['accuracy'])
print("\n=== Phase-2: fine-tuning ===")
model.fit(train_gen, validation_data=val_gen,
          epochs=EPOCHS_FINE, verbose=1)

# Save and eval
model.save('hybrid_deepfake_detector.h5')
print("Model saved as hybrid_deepfake_detector.h5")

print("\n=== Test evaluation ===")
p       = model.predict(test_gen)
y_pred  = (p > 0.5).astype(int).flatten()
y_true  = test_gen.classes

print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred):.4f}")
print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
print(f"F1       : {f1_score(y_true, y_pred):.4f}")
print(f"AUC      : {roc_auc_score(y_true, p):.4f}")
print('Confusion matrix:')
print(confusion_matrix(y_true, y_pred))