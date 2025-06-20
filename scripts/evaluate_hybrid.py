import os
import json
import matplotlib.pyplot as plt
import sklearn.metrics as skm
from sklearn.calibration import calibration_curve
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from vit_keras import vit  # custom ViT layers
from tensorflow.keras import mixed_precision

# ─── Mixed precision policy ─────────────────────────────────────────────────
# Match precision used during training
mixed_precision.set_global_policy("mixed_float16")

# ─── Configuration ──────────────────────────────────────────────────────────
# Update these paths before running
MODEL_PATH = 'path/to/hybrid_model.h5'
TEST_DIR   = 'path/to/test'
OUT_DIR    = 'path/to/output'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
THRESH     = 0.5

os.makedirs(OUT_DIR, exist_ok=True)

# ─── Load model ─────────────────────────────────────────────────────────────
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

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
y_pred = (probs >= THRESH).astype(int)

# ─── Compute metrics ────────────────────────────────────────────────────────
metrics = {
    "accuracy":       skm.accuracy_score(y_true, y_pred),
    "precision":      skm.precision_score(y_true, y_pred, zero_division=0),
    "recall":         skm.recall_score(y_true, y_pred, zero_division=0),
    "f1":             skm.f1_score(y_true, y_pred, zero_division=0),
    "roc_auc":        skm.roc_auc_score(y_true, probs),
    "avg_precision":  skm.average_precision_score(y_true, probs),
    "mcc":            skm.matthews_corrcoef(y_true, y_pred),
    "cohen_kappa":    skm.cohen_kappa_score(y_true, y_pred),
    "brier_score":    skm.brier_score_loss(y_true, probs),
}
cm = skm.confusion_matrix(y_true, y_pred, labels=[0, 1])
metrics.update({
    "TN": int(cm[0,0]), "FP": int(cm[0,1]),
    "FN": int(cm[1,0]), "TP": int(cm[1,1]),
})
metrics["classification_report"] = skm.classification_report(
    y_true, y_pred, output_dict=True
)
pt, pp = calibration_curve(y_true, probs, n_bins=10)
metrics["calibration_curve"] = {
    "prob_true": pt.tolist(),
    "prob_pred": pp.tolist()
}

with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# ─── Generate and save plots ───────────────────────────────────────────────
fpr, tpr, _ = skm.roc_curve(y_true, probs)
plt.figure()
plt.plot(fpr, tpr)
plt.title("Hybrid ROC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig(os.path.join(OUT_DIR, "roc_curve.png"))

prec, rec, _ = skm.precision_recall_curve(y_true, probs)
plt.figure()
plt.plot(rec, prec)
plt.title("Hybrid Precision-Recall")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.savefig(os.path.join(OUT_DIR, "pr_curve.png"))

print(f"✅ Evaluation done – outputs in {OUT_DIR}")
