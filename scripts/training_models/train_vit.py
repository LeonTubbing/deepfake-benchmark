"""
Spatio-temporal ViT deepfake detector training script.
Replace 'path/to/train', 'path/to/val', and 'path/to/test' with your data directories.
"""

import os
import time
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Input, TimeDistributed, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from vit_keras import vit


# Configuration

TRAIN_DIR   = '/path/to/train'
VAL_DIR     = '/path/to/val'
TEST_DIR    = '/path/to/test'
IMAGE_SIZE  = 224
BATCH_SIZE  = 4     # adjust batch for memory
TIME_STEPS  = 8     # number of frames per video sample
LR_INIT     = 1e-3
LR_FINE     = 1e-4
EPOCHS_INIT = 5
EPOCHS_FINE = 5
UNFREEZE_N  = 10    # last N transformer blocks to unfreeze


# Video data sequence loader

class VideoSequence(Sequence):
    def __init__(self, video_paths, labels, batch_size, time_steps,
                 image_size, preprocess_fn, augment=False, shuffle=True):
        self.video_paths = video_paths
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.image_size = image_size
        self.preprocess_fn = preprocess_fn
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(self.video_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.video_paths) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_paths = [self.video_paths[i] for i in batch_idx]
        batch_labels = self.labels[batch_idx]

        batch_data = []
        for p in batch_paths:
            cap = cv2.VideoCapture(p)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # sample evenly spaced frame indices
            if total < self.time_steps:
                frame_ids = list(range(total)) + [total-1] * (self.time_steps - total)
            else:
                frame_ids = np.linspace(0, total - 1, self.time_steps, dtype=int)

            frames = []
            for fid in frame_ids:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
                ret, frame = cap.read()
                if not ret:
                    frame = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
                frame = cv2.resize(frame, (self.image_size, self.image_size))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.augment:
                    # simple augmentation: horizontal flip and random rotation
                    if np.random.rand() < 0.5:
                        frame = cv2.flip(frame, 1)
                    angle = np.random.uniform(-15, 15)
                    M = cv2.getRotationMatrix2D((self.image_size/2, self.image_size/2), angle, 1)
                    frame = cv2.warpAffine(frame, M, (self.image_size, self.image_size))
                frame = self.preprocess_fn(frame)
                frames.append(frame)
            cap.release()
            batch_data.append(np.stack(frames, axis=0))

        X = np.stack(batch_data, axis=0)
        y = batch_labels.astype(np.float32)
        return X, y


# Helpers to list videos and labels

def list_videos_and_labels(base_dir):
    classes = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    label_map = {cls: i for i, cls in enumerate(classes)}
    paths, labels = [], []
    for cls in classes:
        cls_dir = os.path.join(base_dir, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.mp4', '.avi', '.mov')):
                paths.append(os.path.join(cls_dir, fname))
                labels.append(label_map[cls])
    return paths, labels

# Prepare sequences\ ntrain_paths, train_labels = list_videos_and_labels(TRAIN_DIR)
val_paths, val_labels     = list_videos_and_labels(VAL_DIR)
test_paths, test_labels   = list_videos_and_labels(TEST_DIR)

train_seq = VideoSequence(train_paths, train_labels, BATCH_SIZE, TIME_STEPS,
                          IMAGE_SIZE, preprocess_input, augment=True)
val_seq   = VideoSequence(val_paths, val_labels, BATCH_SIZE, TIME_STEPS,
                          IMAGE_SIZE, preprocess_input, augment=False)
test_seq  = VideoSequence(test_paths, test_labels, BATCH_SIZE, TIME_STEPS,
                          IMAGE_SIZE, preprocess_input, augment=False, shuffle=False)


# Build spatio-temporal ViT model

# base ViT processes per-frame
base_vit = vit.vit_b16(
    image_size=IMAGE_SIZE,
    pretrained=True,
    include_top=False,    # outputs [batch, 768]
    pretrained_top=False
)

# wrap in TimeDistributed and pool over time
input_tensor = Input(shape=(TIME_STEPS, IMAGE_SIZE, IMAGE_SIZE, 3))
vit_td = TimeDistributed(base_vit)(input_tensor)  # [batch, T, 768]
time_pooled = GlobalAveragePooling1D()(vit_td)  # [batch, 768]
output = Dense(1, activation='sigmoid')(time_pooled)
model = Model(inputs=input_tensor, outputs=output)

# Phase 1: freeze all base_vit layers
for layer in base_vit.layers:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=LR_INIT),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n=== Starting Phase 1: Initial Training ===")
start = time.time()
model.fit(
    train_seq,
    validation_data=val_seq,
    epochs=EPOCHS_INIT,
    verbose=1
)
print(f"Phase 1 took {(time.time() - start)/60:.2f} minutes\n")

# Phase 2: fine-tune last UNFREEZE_N transformer blocks
for layer in base_vit.layers[-UNFREEZE_N:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=LR_FINE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("=== Starting Phase 2: Fine-Tuning ===")
start = time.time()
model.fit(
    train_seq,
    validation_data=val_seq,
    epochs=EPOCHS_FINE,
    verbose=1
)
print(f"Phase 2 took {(time.time() - start)/60:.2f} minutes\n")

# Save the spatio-temporal model
model.save('vit_spatiotemporal_detector.h5')
print("Saved trained spatio-temporal model to vit_spatiotemporal_detector.h5\n")


# Evaluation on test set

print("=== Evaluating on Test Set ===")
pred_prob   = model.predict(test_seq)
pred_labels = (pred_prob > 0.5).astype(int).flatten()
true_labels = np.array(test_labels)

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
