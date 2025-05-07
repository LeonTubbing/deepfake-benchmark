
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
import time
from tqdm import tqdm

# Define directory paths
train_dir = 'give the pathname to your directory'
val_dir = 'give the pathname to your directory'
test_dir = 'give the pathname to your directory'

# Create ImageDataGenerators for training, validation, and testing
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # To maintain order for evaluation
)

# Load the pre-trained Xception model without the top layer
base_model = Xception(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base_model layers initially
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Phase 1: Initial Training
print("\n Starting initial training...")
start_time = time.time()
history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,
    verbose=1
)
elapsed_time = (time.time() - start_time) / 60
print(f" Initial training took {elapsed_time:.2f} minutes.")

# Fine-tuning: Unfreeze the last 20 layers of the base model
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Recompile with lower learning rate
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Phase 2: Fine-Tuning
print("\n Starting fine-tuning (last 20 layers)...")
start_time = time.time()
history2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,
    verbose=1
)
elapsed_time = (time.time() - start_time) / 60
print(f" Fine-tuning took {elapsed_time:.2f} minutes.")

# Save the trained model
model.save('deepfake_detection_model.h5')

# Evaluation on the test set
print("\n Evaluating on test set...")
predictions_prob = model.predict(test_generator)
predictions_binary = (predictions_prob > 0.5).astype(int).flatten()
true_labels = test_generator.classes

# Compute evaluation metrics
accuracy = accuracy_score(true_labels, predictions_binary)
precision = precision_score(true_labels, predictions_binary)
recall = recall_score(true_labels, predictions_binary)
f1 = f1_score(true_labels, predictions_binary)
auc = roc_auc_score(true_labels, predictions_prob)
cm = confusion_matrix(true_labels, predictions_binary)

print("\n Test Metrics:")
print("Test Accuracy:", accuracy)
print("Test Precision:", precision)
print("Test Recall:", recall)
print("Test F1 Score:", f1)
print("Test AUC:", auc)
print("Confusion Matrix:")
print(cm)