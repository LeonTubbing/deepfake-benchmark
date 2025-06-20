import os
import json
import numpy as np
import sklearn.metrics as skm
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# ─── Configuration ──────────────────────────────────────────────────────────
# Set these to your model file, test set and where to save outputs
MODEL_WEIGHTS = 'path/to/mobilenetv3_weights.h5'
TEST_DIR      = 'path/to/test'
OUTPUT_DIR    = 'path/to/output'
IMAGE_SIZE    = (224, 224)
BATCH_SIZE    = 32
THRESHOLD     = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Build model architecture ───────────────────────────────────────────────
inp  = Input(shape=(*IMAGE_SIZE, 3))
base = MobileNetV3Large(weights='imagenet', include_top=False, input_tensor=inp)
x    = GlobalAveragePooling2D()(base.output)
x    = Dense(256, activation='relu')(x)
x    = Dropout(0.5)(x)
out  = Dense(1, activation='sigmoid')(x)
model = Model(inputs=inp, outputs=out)

# ─── Load trained weights ───────────────────────────────────────────────────
model.load_weights(MODEL_WEIGHTS)

# ─── Prepare test data ──────────────────────────────────────────────────────
test_gen = ImageDataGenerator(preprocessing_function=preprocess_input) \
    .flow_from_directory(
        TEST_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

# ─── Run inference ──────────────────────────────────────────────────────────
y_true = test_gen.classes
probs  = model.predict(test_gen, verbose=1).flatten()
y_pred = (probs >= THRESHOLD).astype(int)

# ─── Compute metrics ────────────────────────────────────────────────────────
metrics = {
    'accuracy':       skm.accuracy_score(y_true, y_pred),
    'precision':      skm.precision_score(y_true, y_pred, zero_division=0),
    'recall':         skm.recall_score(y_true, y_pred, zero_division=0),
    'f1':             skm.f1_score(y_true, y_pred, zero_division=0),
    'roc_auc':        skm.roc_auc_score(y_true, probs),
    'avg_precision':  skm.average_precision_score(y_true, probs),
    'mcc':            skm.matthews_corrcoef(y_true, y_pred),
    'cohen_kappa':    skm.cohen_kappa_score(y_true, y_pred),
    'brier_score':    skm.brier_score_loss(y_true, probs),
}
cm = skm.confusion_matrix(y_true, y_pred, labels=[0,1])
metrics.update({
    'TN': int(cm[0,0]), 'FP': int(cm[0,1]),
    'FN': int(cm[1,0]), 'TP': int(cm[1,1]),
})
metrics['classification_report'] = skm.classification_report(
    y_true, y_pred, output_dict=True
)
prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10)
metrics['calibration_curve'] = {
    'prob_true': prob_true.tolist(),
    'prob_pred': prob_pred.tolist()
}

with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

# ─── Generate and save plots ───────────────────────────────────────────────
fpr, tpr, _ = skm.roc_curve(y_true, probs)
plt.figure()
plt.plot(fpr, tpr)
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve.png'))

prec, rec, _ = skm.precision_recall_curve(y_true, probs)
plt.figure()
plt.plot(rec, prec)
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig(os.path.join(OUTPUT_DIR, 'pr_curve.png'))

print(f"✅ Evaluation done – outputs in {OUTPUT_DIR}")
